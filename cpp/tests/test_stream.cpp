#include <gtest/gtest.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <random>
#include <sstream>
#include <thread>
#include <vector>
#include "qre/stream/feed.hpp"
#include "qre/stream/incremental.hpp"
#include "qre/stream/latency.hpp"
#include "qre/stream/pipeline.hpp"
#include "qre/stream/queue.hpp"
#include "qre/stream/websocket.hpp"

using namespace qre;
using namespace qre::stream;

namespace{
//Same fixture as python_reference/tests/test_stream.py
struct Fx{ Eigen::VectorXd shares, prices; Eigen::MatrixXd cov, r; };
Fx fixture(){
    const int t = 250, n = 4;
    Fx f;
    f.r.resize(t, n);
    for(int i = 0; i < t; ++i)
        for(int j = 0; j < n; ++j){
            f.r(i, j) = 0.01 * std::sin(0.37 * i * (j + 1) + j) + 0.004 * std::cos(1.3 * i + 2.0 * j);
            if(i % 37 == 0) f.r(i, j) -= 0.03 * (1 + j) / n;
        }
    f.shares.resize(n); f.shares << 100.0, -40.0, 250.0, 60.0;
    f.prices.resize(n); f.prices << 180.0, 95.0, 42.0, 310.0;
    f.cov = linalg::sample_cov(f.r);
    return f;
}
std::vector<std::pair<int, double>> ticks(const Eigen::VectorXd& p, int k_max = 500){
    std::vector<std::pair<int, double>> out;
    const int n = static_cast<int>(p.size());
    for(int k = 0; k < k_max; ++k) out.push_back({k % n, p(k % n) * (1.0 + 0.03 * std::sin(0.7 * k + k % n))});
    return out;
}
std::string hex(const std::array<std::uint8_t, 20>& d){
    static const char* h = "0123456789abcdef";
    std::string s;
    for(auto b : d){ s += h[b >> 4]; s += h[b & 15]; }
    return s;
}
}

// ------------------------------------------------------------------------------------------------ queues
TEST(Queue, SpscFifoAndCapacity){
    SpscQueue<int> q(4);
    EXPECT_THROW(SpscQueue<int>(3), InvalidInputError);
    for(int i = 0; i < 4; ++i) EXPECT_TRUE(q.try_push(i));
    EXPECT_FALSE(q.try_push(99));                       //full
    int v = -1;
    for(int i = 0; i < 4; ++i){ ASSERT_TRUE(q.try_pop(v)); EXPECT_EQ(v, i); }
    EXPECT_FALSE(q.try_pop(v));                         //empty
    for(int round = 0; round < 10; ++round){            //indices wrap many times
        EXPECT_TRUE(q.try_push(round));
        ASSERT_TRUE(q.try_pop(v));
        EXPECT_EQ(v, round);
    }
}

template<typename Q>
void two_threads_in_order(){
    Q q(1024);
    const std::uint64_t n = 2'000'000;
    std::atomic<bool> ok{true};
    std::thread consumer([&]{
        std::uint64_t expect = 0, v = 0;
        while(expect < n){
            if(q.try_pop(v)){
                if(v != expect) ok = false;
                ++expect;
            }
        }
    });
    for(std::uint64_t i = 0; i < n; ++i) while(!q.try_push(i)) {}
    consumer.join();
    EXPECT_TRUE(ok);
}
TEST(Queue, SpscTwoThreadsKeepOrder){ two_threads_in_order<SpscQueue<std::uint64_t>>(); }
TEST(Queue, MutexTwoThreadsKeepOrder){ two_threads_in_order<MutexQueue<std::uint64_t>>(); }

// ------------------------------------------------------------------------------------------------ histogram
TEST(Latency, HistogramPercentilesWithinBucketResolution){
    LatencyHistogram h;
    for(std::int64_t v = 1; v <= 100000; ++v) h.record(v);
    EXPECT_EQ(h.count(), 100000u);
    EXPECT_EQ(h.max(), 100000u);
    EXPECT_NEAR(h.mean(), 50000.5, 1e-9);
    for(double p : {50.0, 90.0, 99.0, 99.9}) EXPECT_NEAR(h.percentile(p) / (p * 1000.0), 1.0, 1.0 / 32);
    for(std::uint64_t v : {0ull, 31ull, 32ull, 1000ull, 123456789ull}){
        const auto [lo, hi] = LatencyHistogram::bounds(LatencyHistogram::index(v));
        EXPECT_LE(lo, v);
        EXPECT_GE(hi, v);
    }
    LatencyHistogram g;
    g.record(5);
    h.merge(g);
    EXPECT_EQ(h.count(), 100001u);
}

// ------------------------------------------------------------------------------------------------ incremental risk
TEST(Incremental, EqualsFullRecomputeAfterEveryTick){
    const Fx f = fixture();
    IncrementalRisk risk(f.shares, f.prices, f.cov, f.r, 0.99);
    Eigen::VectorXd p = f.prices;
    std::vector<double> scratch;
    const double v0 = f.shares.dot(f.prices);
    int k = 0;
    for(auto [i, price] : ticks(f.prices)){
        risk.on_tick(i, price);
        p(i) = price;
        if(k++ % 50 != 0) continue;
        const RiskSnapshot a = risk.snapshot();
        const RiskSnapshot b = full_recompute(f.shares, p, f.cov, f.r, 0.99, v0, scratch);
        EXPECT_NEAR(a.value / b.value, 1.0, 1e-13);
        EXPECT_NEAR(a.sigma / b.sigma, 1.0, 1e-12);
        EXPECT_NEAR(a.var_historical / b.var_historical, 1.0, 1e-12);
        EXPECT_NEAR(a.es_historical / b.es_historical, 1.0, 1e-12);
        EXPECT_NEAR(a.component_var.sum() / a.var_parametric, 1.0, 1e-12);
    }
    const auto d = risk.drift();
    EXPECT_LT(std::max({d.variance, d.exposure, d.scenario_loss}), 1e-12);
    risk.resync();
    EXPECT_EQ(risk.drift().variance, 0.0);
    EXPECT_THROW(risk.on_tick(0, -1.0), InvalidInputError);
    EXPECT_THROW(IncrementalRisk(f.shares, f.prices, f.cov, f.r.topRows(10)), InvalidInputError);
}

TEST(Incremental, MatchesPythonReference){
    const Fx f = fixture();
    IncrementalRisk risk(f.shares, f.prices, f.cov, f.r, 0.975);
    for(auto [i, price] : ticks(f.prices)) risk.on_tick(i, price);
    const RiskSnapshot s = risk.snapshot();
    EXPECT_NEAR(s.value, 43798.72387759365, 1e-9);
    EXPECT_NEAR(s.pnl, 498.7238775936494, 1e-9);
    EXPECT_NEAR(s.sigma, 266.01768306576565, 1e-10);
    EXPECT_NEAR(s.var_parametric, 521.3850780596913, 1e-7);      //C++ normal quantile is a Brent root
    EXPECT_NEAR(s.es_parametric, 621.8968822460976, 1e-7);
    EXPECT_NEAR(s.var_historical, 633.9301400698679, 1e-9);
    EXPECT_NEAR(s.es_historical, 852.8840476556697, 1e-9);
    const double py[] = {181.877293533477, 3.462184199621236, 73.20629523668005, 262.8393050899131};
    for(int j = 0; j < 4; ++j) EXPECT_NEAR(s.component_var(j), py[j], 1e-7);
}

TEST(Incremental, QuantileBySelectionMatchesSorting){
    std::mt19937_64 g(5);
    std::student_t_distribution<double> t(4.0);
    std::vector<double> x(257);
    for(auto& v : x) v = t(g);
    for(double q : {0.5, 0.95, 0.975, 0.99, 0.999}){
        std::vector<double> a = x, b = x;
        std::sort(b.begin(), b.end());
        const double h = 256.0 * q;
        const auto lo = static_cast<std::size_t>(h);
        const double expect = b[lo] + (h - static_cast<double>(lo)) * (b[std::min<std::size_t>(lo + 1, 256)] - b[lo]);
        EXPECT_DOUBLE_EQ(quantile_linear(a, q), expect);
    }
}

TEST(Feed, SyntheticReproducesCovariance){
    Eigen::Matrix2d cov;
    cov << 4e-4, 1.2e-4, 1.2e-4, 1e-4;
    SyntheticFeed feed(Eigen::Vector2d(100.0, 50.0), cov, 1.0, 7);       //one simulated day per step
    std::vector<double> a, b;
    double pa = 100.0, pb = 50.0;
    for(int k = 0; k < 40000; ++k){
        const Tick t1 = feed.next(), t2 = feed.next();
        ASSERT_EQ(t1.asset, 0u);
        ASSERT_EQ(t2.asset, 1u);
        a.push_back(std::log(t1.price / pa)); b.push_back(std::log(t2.price / pb));
        pa = t1.price; pb = t2.price;
    }
    auto mean = [](const std::vector<double>& v){ return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); };
    const double ma = mean(a), mb = mean(b);
    double vaa = 0, vbb = 0, vab = 0;
    for(std::size_t i = 0; i < a.size(); ++i){ vaa += (a[i] - ma) * (a[i] - ma); vbb += (b[i] - mb) * (b[i] - mb); vab += (a[i] - ma) * (b[i] - mb); }
    const double n = static_cast<double>(a.size() - 1);
    EXPECT_NEAR(vaa / n / 4e-4, 1.0, 0.03);
    EXPECT_NEAR(vbb / n / 1e-4, 1.0, 0.03);
    EXPECT_NEAR(vab / std::sqrt(vaa * vbb), 0.6, 0.02);
}

// ------------------------------------------------------------------------------------------------ WebSocket
TEST(WebSocket, Sha1AndBase64TestVectors){
    EXPECT_EQ(hex(ws::sha1("")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    EXPECT_EQ(hex(ws::sha1("abc")), "a9993e364706816aba3e25717850c26c9cd0d89d");
    EXPECT_EQ(hex(ws::sha1("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")), "84983e441c3bd26ebaae4aa1f95129e5e54670f1");
    const std::string a(1000000, 'a');
    EXPECT_EQ(hex(ws::sha1(a)), "34aa973cd4c4daa4f61eeb2bdbad27316534016f");
    const std::uint8_t m[] = {'M', 'a', 'n'};
    EXPECT_EQ(ws::base64(m, 3), "TWFu");
    EXPECT_EQ(ws::base64(m, 2), "TWE=");
    EXPECT_EQ(ws::base64(m, 1), "TQ==");
    EXPECT_EQ(ws::accept_key("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");    //RFC 6455 section 1.3
}

TEST(WebSocket, FramesRoundTripAtEveryLengthEncoding){
    for(std::size_t n : {0u, 5u, 125u, 126u, 65535u, 65536u, 200000u}){
        const std::string payload(n, 'x');
        const std::string f = ws::encode_frame(payload);
        EXPECT_EQ(static_cast<std::uint8_t>(f[0]), 0x81);
        //mask it as a client would and decode
        std::string c = f.substr(0, 1);
        const std::size_t hdr = n < 126 ? 2 : n <= 65535 ? 4 : 10;
        c += static_cast<char>(static_cast<std::uint8_t>(f[1]) | 0x80);
        c += f.substr(2, hdr - 2);
        const std::uint8_t mask[4] = {1, 2, 3, 4};
        c.append(reinterpret_cast<const char*>(mask), 4);
        for(std::size_t i = 0; i < n; ++i) c += static_cast<char>(payload[i] ^ static_cast<char>(mask[i % 4]));
        if(n > (1u << 20)) continue;
        const auto d = ws::decode_frame(c);
        ASSERT_TRUE(d.has_value());
        EXPECT_EQ(d->payload, payload);
        EXPECT_EQ(d->consumed, c.size());
        EXPECT_FALSE(ws::decode_frame(std::string_view(c).substr(0, c.size() - 1)).has_value());   //incomplete
    }
    EXPECT_THROW(ws::decode_frame(ws::encode_frame("unmasked")), InvalidInputError);
}

TEST(WebSocket, RequestParsingAndOrigins){
    const auto r = ws::parse_request("GET /snapshot?x=1 HTTP/1.1\r\nHost: a\r\nUpgrade:  WebSocket \r\nOrigin: http://localhost:8000\r\n\r\n");
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->path, "/snapshot?x=1");
    EXPECT_EQ(r->header("upgrade"), "WebSocket");
    EXPECT_FALSE(ws::parse_request("GET / HTTP/1.1\r\nHost: a\r\n").has_value());
    EXPECT_TRUE(ws::local_origin(""));
    EXPECT_TRUE(ws::local_origin("http://127.0.0.1:8000"));
    EXPECT_TRUE(ws::local_origin("http://localhost"));
    EXPECT_FALSE(ws::local_origin("http://localhost.evil.com"));
    EXPECT_FALSE(ws::local_origin("https://example.com"));
}

namespace{
int connect_local(int port){
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(static_cast<std::uint16_t>(port));
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if(::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0){ ::close(fd); return -1; }
    timeval tv{2, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    return fd;
}
std::string read_until(int fd, ws::Server& srv, const std::string& marker, std::size_t min_bytes = 0){
    std::string got;
    for(int k = 0; k < 400 && (got.find(marker) == std::string::npos || got.size() < min_bytes); ++k){
        srv.poll_once(5);
        char buf[4096];
        const ssize_t n = ::recv(fd, buf, sizeof buf, MSG_DONTWAIT);
        if(n > 0) got.append(buf, static_cast<std::size_t>(n));
    }
    return got;
}
std::string handler(const ws::Request& r, void*){ return r.path == "/snapshot" ? "{\"ok\":1}" : ""; }
}

TEST(WebSocket, ServerHandshakeBroadcastAndHttp){
    ws::Server srv(0, &handler);                          //port 0: the OS picks a free port
    const int c = connect_local(srv.port());
    ASSERT_GE(c, 0);
    const std::string req = "GET / HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                            "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n";
    ::send(c, req.data(), req.size(), 0);
    const std::string resp = read_until(c, srv, "\r\n\r\n");
    EXPECT_NE(resp.find("101 Switching Protocols"), std::string::npos);
    EXPECT_NE(resp.find("s3pPLMBiTxaQ9kYGzzhZRbK+xOo="), std::string::npos);
    EXPECT_EQ(srv.clients(), 1u);
    srv.broadcast("{\"hello\":1}");
    const std::string frame = read_until(c, srv, "hello");
    EXPECT_EQ(frame, ws::encode_frame("{\"hello\":1}"));
    ::close(c);
    //plain HTTP: the handler's body with 200, unknown paths 404, foreign origins 403
    for(auto [path, origin, code] : {std::tuple{"/snapshot", "", "200"}, std::tuple{"/nope", "", "404"},
                                     std::tuple{"/snapshot", "Origin: https://evil.example\r\n", "403"}}){
        const int h = connect_local(srv.port());
        const std::string get = std::string("GET ") + path + " HTTP/1.1\r\nHost: x\r\n" + origin + "\r\n";
        ::send(h, get.data(), get.size(), 0);
        const std::string r = read_until(h, srv, "\r\n\r\n");
        EXPECT_EQ(r.substr(9, 3), code) << path;
        ::close(h);
    }
    for(int k = 0; k < 20 && srv.clients() > 0; ++k) srv.poll_once(5);
    EXPECT_EQ(srv.clients(), 0u);                         //the closed WebSocket client was removed
}

// ------------------------------------------------------------------------------------------------ pipeline
TEST(StreamPipeline, LinesSourceMatchesDirectComputation){
    const Fx f = fixture();
    data::PriceTable t;
    const std::vector<std::string> names = {"AAA", "BBB", "CCC", "DDD"};
    t.tickers = names;
    t.prices.resize(f.r.rows() + 1, 4);
    t.prices.row(0) = f.prices.transpose();
    for(Eigen::Index i = 0; i < f.r.rows(); ++i){
        t.prices.row(i + 1) = t.prices.row(i).cwiseProduct((1.0 + f.r.row(i).array()).matrix());
        t.dates.push_back("d" + std::to_string(i));
    }
    t.dates.push_back("end");
    PortfolioSetup s = make_setup(t, {{"AAA", 100}, {"BBB", -40}, {"CCC", 250}, {"DDD", 60}}, 250, false, 0.0);
    std::stringstream in;
    in.precision(17);
    in << "ZZZ,1\nnot a tick\n";                          //unknown ticker and junk are skipped
    for(auto [i, price] : ticks(s.prices)) in << names[static_cast<std::size_t>(i)] << "," << price << "\n";
    PipelineOptions o;
    o.source = "lines";
    o.confidence = 0.975;
    o.resync_every = 0;
    Pipeline p(s, o, &in);
    p.start();
    for(int k = 0; k < 500 && !p.finished(); ++k) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ASSERT_TRUE(p.finished());
    IncrementalRisk direct(s.shares, s.prices, s.cov, s.scenarios, 0.975);
    for(auto [i, price] : ticks(s.prices)) direct.on_tick(i, price);
    const std::string j = p.json();
    EXPECT_NE(j.find("\"ticks\":500,"), std::string::npos) << j.substr(0, 120);
    char want[64];
    std::snprintf(want, sizeof want, "\"var_historical\":%.6f", direct.snapshot().var_historical);
    EXPECT_NE(j.find(want), std::string::npos);
    EXPECT_THROW(parse_holdings("AAPL"), InvalidInputError);
    EXPECT_THROW(make_setup(t, {{"XXX", 1}}, 250, false, 0.0), InvalidInputError);
    EXPECT_THROW(make_setup(t, {{"AAA", 1}}, 400, false, 0.0), InvalidInputError);
}
