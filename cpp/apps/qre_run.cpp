//qre_run: the production entry point
//  qre_run --config configs/daily_risk.conf [--threads N] [--output DIR]
//Reads the config, runs the engine (load -> harden -> forecast every model in parallel -> score -> combine ->
//backtest), and writes the report described in qre/engine/report.hpp. Exit code 0 on success, 1 on bad input
//(the message says which line or value), 2 on bad arguments
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include "qre/engine/config.hpp"
#include "qre/engine/engine.hpp"
#include "qre/engine/report.hpp"

namespace {
int usage(){
    std::cerr << "usage: qre_run --config FILE [--threads N] [--output DIR]\n"
                 "  --threads 0 (default) uses every hardware thread; results do not depend on it\n";
    return 2;
}
}

int main(int argc, char** argv){
    std::string config_path, output;
    std::size_t threads = 0;
    for(int i = 1; i < argc; ++i){
        const std::string a = argv[i];
        if((a == "--config" || a == "--threads" || a == "--output") && i + 1 < argc){
            const std::string v = argv[++i];
            if(a == "--config") config_path = v;
            else if(a == "--output") output = v;
            else{
                try{ threads = static_cast<std::size_t>(std::stoul(v)); } catch(const std::exception&){ return usage(); }
            }
        } else return usage();
    }
    if(config_path.empty()) return usage();
    try{
        qre::engine::RunConfig cfg = qre::engine::load_config(config_path);
        if(!output.empty()) cfg.output = output;
        qre::engine::StageTimes times;
        const qre::engine::EngineResult r = qre::engine::run_engine(cfg, threads, &times);
        qre::engine::write_report(r, cfg.output);
        qre::engine::write_run_log(times, cfg.output);
        std::cout << "qre_run: " << r.results.size() << " confidence level(s) x " << cfg.models.size() << " models x "
                  << r.results.front().forecasts.loss.size() + 1 << " days on " << times.threads << " thread(s) in "
                  << (times.load_ms + times.forecast_ms + times.combine_ms) / 1000.0 << " s -> " << cfg.output << "\n";
        for(const auto& cr : r.results){
            std::cout << "  next-day " << 100.0 * cr.confidence << "% VaR " << 100.0 * cr.next_combined.var << "%, ES "
                      << 100.0 * cr.next_combined.es << "% (combined)\n";
        }
    } catch(const std::exception& e){
        std::cerr << "qre_run: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
