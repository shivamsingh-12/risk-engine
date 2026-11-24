//qre_stream: streaming portfolio risk over WebSocket (Phase 22)
//  qre_stream --data FILE --holdings AAPL=100,JPM=50 [--weights --value 100000] [--source synthetic|stdin]
//             [--rate 1000] [--speed 0.02] [--confidence 0.99] [--window 250] [--port 8001] [--publish-hz 10]
//             [--duration SECONDS] [--seed N] [--busy-poll]
//Loads the last `window` days of prices for the holdings, then updates the portfolio's value, P&L and 1-day VaR/ES
//on every price tick and pushes a JSON snapshot to WebSocket clients `publish-hz` times a second.
//  ws://127.0.0.1:PORT/        the stream (one JSON snapshot per message)
//  http://127.0.0.1:PORT/snapshot  the latest snapshot;  /health  {"ok":true}
//Sources: synthetic (a simulated market with the holdings' covariance, `rate` ticks per second, `speed` simulated
//trading days per second) or stdin (lines TICKER,PRICE, e.g. live quotes piped in by the app).
//--busy-poll keeps the risk thread spinning on the queue (lowest latency, one core fully used); by default it
//spins briefly and then sleeps 50 microseconds at a time when no ticks arrive.
//Exit codes: 0 done, 1 bad input, 2 bad arguments.
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include "qre/data/csv_loader.hpp"
#include "qre/stream/pipeline.hpp"
#include "qre/stream/websocket.hpp"

namespace{
std::atomic<bool> g_stop{false};
void on_signal(int){ g_stop = true; }

int usage(const std::string& why = ""){
    if(!why.empty()) std::cerr << "qre_stream: " << why << "\n";
    std::cerr << "usage: qre_stream --data FILE --holdings AAPL=100,JPM=50 [--weights --value V] [--source synthetic|stdin]\n"
                 "                  [--rate TICKS_PER_SEC] [--speed DAYS_PER_SEC] [--confidence C] [--window DAYS]\n"
                 "                  [--port PORT] [--publish-hz HZ] [--duration SECONDS] [--seed N] [--busy-poll]\n";
    return 2;
}

std::string serve(const qre::stream::ws::Request& r, void* ctx){
    auto* p = static_cast<qre::stream::Pipeline*>(ctx);
    const std::string path = r.path.substr(0, r.path.find('?'));
    if(path == "/snapshot") return p->json();
    if(path == "/health") return "{\"ok\":true}";
    return "";
}
}

int main(int argc, char** argv){
    using namespace qre::stream;
    std::string data, holdings, source = "synthetic";
    bool weights = false;
    double value = 100000.0, duration = 0.0;
    int window = 250, port = 8001;
    PipelineOptions opt;
    try{
        for(int i = 1; i < argc; ++i){
            const std::string a = argv[i];
            auto next = [&]() -> std::string{
                if(i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
                return argv[++i];
            };
            if(a == "--data") data = next();
            else if(a == "--holdings") holdings = next();
            else if(a == "--weights") weights = true;
            else if(a == "--value") value = std::stod(next());
            else if(a == "--source") source = next();
            else if(a == "--rate") opt.rate = std::stod(next());
            else if(a == "--speed") opt.speed = std::stod(next());
            else if(a == "--confidence") opt.confidence = std::stod(next());
            else if(a == "--window") window = std::stoi(next());
            else if(a == "--port") port = std::stoi(next());
            else if(a == "--publish-hz") opt.publish_hz = std::stod(next());
            else if(a == "--duration") duration = std::stod(next());
            else if(a == "--seed") opt.seed = std::stoull(next());
            else if(a == "--busy-poll") opt.busy_poll = true;
            else return usage("unknown argument " + a);
        }
    } catch(const std::exception& e){
        return usage(e.what());
    }
    if(data.empty() || holdings.empty()) return usage("--data and --holdings are required");
    if(source != "synthetic" && source != "stdin") return usage("--source must be synthetic or stdin");
    if(!(opt.rate >= 0.0) || !(opt.speed > 0.0) || window < 20 || port < 0 || port > 65535 || !(value > 0.0))
        return usage("rate >= 0, speed > 0, window >= 20, a valid port and a positive value are required");
    opt.source = source == "stdin" ? "lines" : "synthetic";

    try{
        const auto table = qre::data::load_price_csv(data);
        PortfolioSetup setup = make_setup(table, parse_holdings(holdings), window, weights, value);
        Pipeline pipe(std::move(setup), opt, source == "stdin" ? &std::cin : nullptr);
        ws::Server server(port, &serve, &pipe);
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        std::cout << "qre_stream: listening on 127.0.0.1:" << server.port() << " (" << source << ", "
                  << pipe.setup().tickers.size() << " holdings, window " << pipe.setup().first_date << " to "
                  << pipe.setup().last_date << ")" << std::endl;
        pipe.start();
        const auto t0 = std::chrono::steady_clock::now();
        const auto every = std::chrono::duration<double>(1.0 / opt.publish_hz);
        auto next_push = t0;
        std::string last;
        while(!g_stop){
            server.poll_once(5);
            const auto now = std::chrono::steady_clock::now();
            if(now >= next_push){
                std::string j = pipe.json();
                if(j != last){ server.broadcast(j); last = std::move(j); }
                next_push = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(every);
            }
            if(duration > 0.0 && now - t0 >= std::chrono::duration<double>(duration)) break;
            if(pipe.finished()){ server.broadcast(pipe.json()); break; }
        }
        pipe.stop();
        std::cout << "qre_stream: stopped. " << pipe.json().substr(0, 200) << "..." << std::endl;
    } catch(const std::exception& e){
        std::cerr << "qre_stream: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
