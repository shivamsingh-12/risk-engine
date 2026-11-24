#pragma once
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include "qre/core/exceptions.hpp"

//Phase 22: a small WebSocket (RFC 6455) and HTTP server over POSIX sockets, written from scratch for qre_stream.
//It does what the risk stream needs and no more:
//  - listens on 127.0.0.1 only and accepts browser connections only from http://127.0.0.1 or http://localhost
//    origins (any port), so other web pages cannot read the stream
//  - upgrades a GET with "Upgrade: websocket" (the handshake answer is base64(SHA-1(key + RFC GUID)))
//  - answers other GETs through a handler (qre_stream serves /snapshot as JSON); unknown paths get 404
//  - pushes text frames to every client; reads client frames only to answer ping and close
//  - never blocks the caller: sockets are non-blocking, and a client that cannot keep up (its send buffer is
//    full) is dropped rather than allowed to slow the pipeline
//SHA-1 and base64 are implemented here for the handshake; SHA-1 is used only because the protocol requires it.
namespace qre::stream::ws{
    // ---------------------------------------------------------------------------------------- SHA-1, base64
    inline std::array<std::uint8_t, 20> sha1(std::string_view msg){
        std::uint32_t h[5] = {0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u};
        std::vector<std::uint8_t> m(msg.begin(), msg.end());
        const std::uint64_t bits = static_cast<std::uint64_t>(msg.size()) * 8u;
        m.push_back(0x80);
        while(m.size() % 64 != 56) m.push_back(0);
        for(int i = 7; i >= 0; --i) m.push_back(static_cast<std::uint8_t>(bits >> (8 * i)));
        auto rol = [](std::uint32_t x, int s){ return (x << s) | (x >> (32 - s)); };
        for(std::size_t off = 0; off < m.size(); off += 64){
            std::uint32_t w[80];
            for(int i = 0; i < 16; ++i)
                w[i] = (std::uint32_t{m[off + 4 * static_cast<std::size_t>(i)]} << 24) | (std::uint32_t{m[off + 4 * static_cast<std::size_t>(i) + 1]} << 16) |
                       (std::uint32_t{m[off + 4 * static_cast<std::size_t>(i) + 2]} << 8) | std::uint32_t{m[off + 4 * static_cast<std::size_t>(i) + 3]};
            for(int i = 16; i < 80; ++i) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
            std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
            for(int i = 0; i < 80; ++i){
                std::uint32_t f, k;
                if(i < 20){ f = (b & c) | (~b & d); k = 0x5A827999u; }
                else if(i < 40){ f = b ^ c ^ d; k = 0x6ED9EBA1u; }
                else if(i < 60){ f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDCu; }
                else { f = b ^ c ^ d; k = 0xCA62C1D6u; }
                const std::uint32_t t = rol(a, 5) + f + e + k + w[i];
                e = d; d = c; c = rol(b, 30); b = a; a = t;
            }
            h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
        }
        std::array<std::uint8_t, 20> out{};
        for(int i = 0; i < 5; ++i) for(int j = 0; j < 4; ++j) out[static_cast<std::size_t>(4 * i + j)] = static_cast<std::uint8_t>(h[i] >> (24 - 8 * j));
        return out;
    }

    inline std::string base64(const std::uint8_t* data, std::size_t n){
        static constexpr char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string out;
        for(std::size_t i = 0; i < n; i += 3){
            const std::uint32_t v = (std::uint32_t{data[i]} << 16) | (i + 1 < n ? std::uint32_t{data[i + 1]} << 8 : 0u) | (i + 2 < n ? std::uint32_t{data[i + 2]} : 0u);
            out += tbl[(v >> 18) & 63];
            out += tbl[(v >> 12) & 63];
            out += i + 1 < n ? tbl[(v >> 6) & 63] : '=';
            out += i + 2 < n ? tbl[v & 63] : '=';
        }
        return out;
    }

    inline std::string accept_key(std::string_view client_key){
        const auto d = sha1(std::string(client_key) + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
        return base64(d.data(), d.size());
    }

    // ---------------------------------------------------------------------------------------- frames
    //server -> client frame: FIN set, no mask (servers must not mask)
    inline std::string encode_frame(std::string_view payload, std::uint8_t opcode = 0x1){
        std::string f;
        f += static_cast<char>(0x80 | opcode);
        const std::uint64_t n = payload.size();
        if(n < 126) f += static_cast<char>(n);
        else if(n <= 0xFFFF){ f += static_cast<char>(126); f += static_cast<char>(n >> 8); f += static_cast<char>(n & 0xFF); }
        else { f += static_cast<char>(127); for(int i = 7; i >= 0; --i) f += static_cast<char>((n >> (8 * i)) & 0xFF); }
        f.append(payload);
        return f;
    }

    struct Frame{ std::uint8_t opcode; std::string payload; std::size_t consumed; };

    //client -> server frame (must be masked); nullopt until the whole frame has arrived. Throws on a protocol error
    inline std::optional<Frame> decode_frame(std::string_view buf){
        if(buf.size() < 2) return std::nullopt;
        const auto b0 = static_cast<std::uint8_t>(buf[0]), b1 = static_cast<std::uint8_t>(buf[1]);
        if(!(b1 & 0x80)) throw InvalidInputError("client frames must be masked");
        std::uint64_t len = b1 & 0x7F;
        std::size_t pos = 2;
        if(len == 126){
            if(buf.size() < 4) return std::nullopt;
            len = (std::uint64_t{static_cast<std::uint8_t>(buf[2])} << 8) | static_cast<std::uint8_t>(buf[3]);
            pos = 4;
        } else if(len == 127){
            if(buf.size() < 10) return std::nullopt;
            len = 0;
            for(int i = 0; i < 8; ++i) len = (len << 8) | static_cast<std::uint8_t>(buf[2 + static_cast<std::size_t>(i)]);
            pos = 10;
        }
        if(len > (1u << 20)) throw InvalidInputError("client frame too large");
        if(buf.size() < pos + 4 + len) return std::nullopt;
        const auto* mask = reinterpret_cast<const std::uint8_t*>(buf.data() + pos);
        pos += 4;
        std::string payload(len, '\0');
        for(std::uint64_t i = 0; i < len; ++i) payload[i] = static_cast<char>(static_cast<std::uint8_t>(buf[pos + i]) ^ mask[i % 4]);
        return Frame{static_cast<std::uint8_t>(b0 & 0x0F), std::move(payload), pos + len};
    }

    // ---------------------------------------------------------------------------------------- HTTP request
    struct Request{
        std::string method, path;
        std::vector<std::pair<std::string, std::string>> headers;   //names lower-cased
        std::string header(std::string_view name) const{
            for(const auto& [k, v] : headers) if(k == name) return v;
            return {};
        }
    };

    inline std::optional<Request> parse_request(std::string_view raw){
        const auto end = raw.find("\r\n\r\n");
        if(end == std::string_view::npos) return std::nullopt;
        Request r;
        std::size_t line_end = raw.find("\r\n");
        const std::string_view first = raw.substr(0, line_end);
        const auto s1 = first.find(' '), s2 = first.find(' ', s1 + 1);
        if(s1 == std::string_view::npos || s2 == std::string_view::npos) throw InvalidInputError("bad request line");
        r.method = std::string(first.substr(0, s1));
        r.path = std::string(first.substr(s1 + 1, s2 - s1 - 1));
        std::size_t pos = line_end + 2;
        while(pos < end){
            const auto e = raw.find("\r\n", pos);
            const std::string_view line = raw.substr(pos, e - pos);
            const auto c = line.find(':');
            if(c != std::string_view::npos){
                std::string k(line.substr(0, c));
                for(auto& ch : k) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
                std::string_view v = line.substr(c + 1);
                while(!v.empty() && v.front() == ' ') v.remove_prefix(1);
                while(!v.empty() && v.back() == ' ') v.remove_suffix(1);
                r.headers.emplace_back(std::move(k), std::string(v));
            }
            pos = e + 2;
        }
        return r;
    }

    //browser pages served from this machine only; non-browser clients send no Origin
    inline bool local_origin(std::string_view origin){
        if(origin.empty()) return true;
        for(std::string_view p : {"http://127.0.0.1", "http://localhost"}){
            if(origin.substr(0, p.size()) == p && (origin.size() == p.size() || origin[p.size()] == ':')) return true;
        }
        return false;
    }

    inline bool contains_token(std::string v, std::string_view token){
        for(auto& ch : v) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        return v.find(token) != std::string::npos;
    }

    // ---------------------------------------------------------------------------------------- server
    class Server{
    public:
        using HttpHandler = std::string (*)(const Request&, void*);     //returns the JSON body, or "" for 404

        explicit Server(int port, HttpHandler handler = nullptr, void* ctx = nullptr) : handler_(handler), ctx_(ctx){
            fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
            if(fd_ < 0) throw InvalidInputError(std::string("socket: ") + std::strerror(errno));
            int one = 1;
            ::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
            sockaddr_in a{};
            a.sin_family = AF_INET;
            a.sin_port = htons(static_cast<std::uint16_t>(port));
            a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            if(::bind(fd_, reinterpret_cast<sockaddr*>(&a), sizeof a) < 0 || ::listen(fd_, 16) < 0){
                const std::string err = std::strerror(errno);
                ::close(fd_);
                throw InvalidInputError("cannot listen on 127.0.0.1:" + std::to_string(port) + ": " + err);
            }
            set_nonblocking(fd_);
            socklen_t len = sizeof a;
            ::getsockname(fd_, reinterpret_cast<sockaddr*>(&a), &len);
            port_ = ntohs(a.sin_port);
        }
        ~Server(){
            for(auto& c : conns_) ::close(c.fd);
            if(fd_ >= 0) ::close(fd_);
        }
        Server(const Server&) = delete;
        Server& operator=(const Server&) = delete;

        int port() const{ return port_; }
        std::size_t clients() const{
            std::size_t n = 0;
            for(const auto& c : conns_) n += c.websocket;
            return n;
        }

        //accept connections, read requests and client frames; waits up to timeout_ms for activity
        void poll_once(int timeout_ms){
            std::vector<pollfd> fds;
            fds.push_back({fd_, POLLIN, 0});
            for(const auto& c : conns_) fds.push_back({c.fd, POLLIN, 0});
            if(::poll(fds.data(), static_cast<nfds_t>(fds.size()), timeout_ms) <= 0) return;
            if(fds[0].revents & POLLIN){
                for(;;){
                    const int cfd = ::accept(fd_, nullptr, nullptr);
                    if(cfd < 0) break;
                    set_nonblocking(cfd);
                    int one = 1;
                    ::setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
#ifdef SO_NOSIGPIPE
                    ::setsockopt(cfd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
                    conns_.push_back({cfd, false, {}, false});
                }
            }
            for(std::size_t i = 1; i < fds.size(); ++i){
                if(fds[i].revents & (POLLIN | POLLHUP | POLLERR)) read_from(conns_[i - 1]);
            }
            reap();
        }

        //push a text message to every WebSocket client; drops clients that cannot keep up
        void broadcast(std::string_view text){
            const std::string frame = encode_frame(text);
            for(auto& c : conns_) if(c.websocket && !c.dead) send_all(c, frame);
            reap();
        }

    private:
        struct Conn{ int fd; bool websocket; std::string in; bool dead; };

        static void set_nonblocking(int fd){ ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL, 0) | O_NONBLOCK); }

        void send_all(Conn& c, std::string_view data){
            int flags = 0;
#ifdef MSG_NOSIGNAL
            flags = MSG_NOSIGNAL;
#endif
            std::size_t off = 0;
            while(off < data.size()){
                const ssize_t k = ::send(c.fd, data.data() + off, data.size() - off, flags);
                if(k <= 0){ c.dead = true; return; }     //error, closed, or send buffer full (slow client): drop
                off += static_cast<std::size_t>(k);
            }
        }

        void read_from(Conn& c){
            char buf[4096];
            for(;;){
                const ssize_t k = ::recv(c.fd, buf, sizeof buf, 0);
                if(k > 0){ c.in.append(buf, static_cast<std::size_t>(k)); if(c.in.size() > 65536){ c.dead = true; return; } continue; }
                if(k == 0){ c.dead = true; return; }
                if(errno == EAGAIN || errno == EWOULDBLOCK) break;
                c.dead = true;
                return;
            }
            try{
                if(!c.websocket) handle_http(c);
                else handle_frames(c);
            } catch(const std::exception&){
                c.dead = true;
            }
        }

        void handle_http(Conn& c){
            const auto req = parse_request(c.in);
            if(!req) return;
            const std::string origin = req->header("origin");
            if(!local_origin(origin)){ respond(c, 403, "{\"error\":\"origin not allowed\"}", origin); return; }
            if(req->method == "GET" && contains_token(req->header("upgrade"), "websocket")){
                const std::string key = req->header("sec-websocket-key");
                if(key.empty()){ respond(c, 400, "{\"error\":\"missing Sec-WebSocket-Key\"}", origin); return; }
                send_all(c, "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                            "Sec-WebSocket-Accept: " + accept_key(key) + "\r\n\r\n");
                c.websocket = true;
                c.in.clear();
                return;
            }
            if(req->method == "OPTIONS"){ respond(c, 204, "", origin); return; }
            const std::string body = (req->method == "GET" && handler_) ? handler_(*req, ctx_) : std::string{};
            if(body.empty()) respond(c, 404, "{\"error\":\"not found\"}", origin);
            else respond(c, 200, body, origin);
        }

        void respond(Conn& c, int code, const std::string& body, const std::string& origin){
            const char* reason = code == 200 ? "OK" : code == 204 ? "No Content" : code == 400 ? "Bad Request" : code == 403 ? "Forbidden" : "Not Found";
            std::string h = "HTTP/1.1 " + std::to_string(code) + " " + reason + "\r\nContent-Type: application/json\r\n"
                            "Cache-Control: no-store\r\nConnection: close\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
            if(!origin.empty() && local_origin(origin)) h += "Access-Control-Allow-Origin: " + origin + "\r\nVary: Origin\r\n";
            send_all(c, h + "\r\n" + body);
            c.dead = true;                                                       //one request per connection
        }

        void handle_frames(Conn& c){
            while(auto f = decode_frame(c.in)){
                c.in.erase(0, f->consumed);
                if(f->opcode == 0x8){ send_all(c, encode_frame(f->payload.substr(0, 2), 0x8)); c.dead = true; return; }   //close
                if(f->opcode == 0x9) send_all(c, encode_frame(f->payload, 0xA));                                        //ping -> pong
            }
        }

        void reap(){
            for(auto it = conns_.begin(); it != conns_.end();){
                if(it->dead){ ::close(it->fd); it = conns_.erase(it); } else ++it;
            }
        }

        int fd_ = -1, port_ = 0;
        HttpHandler handler_;
        void* ctx_;
        std::vector<Conn> conns_;
    };
}
