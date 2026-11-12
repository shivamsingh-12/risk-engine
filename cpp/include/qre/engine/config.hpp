#pragma once
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "qre/core/exceptions.hpp"
#include "qre/risk/adaptive.hpp"
#include "qre/risk/backtest.hpp"
//Phase 14: run configuration for the production engine
//Plain "key = value" lines, '#' starts a comment, unknown keys and malformed values are errors (fail loudly).
//The same file is read by the Python reference (qre/engine/config.py), so both engines run the same job.
//  data        = path to the price CSV (relative paths are relative to the working directory)
//  weights     = equal | TICKER=w, TICKER=w, ...   (unlisted tickers get 0, weights must sum to 1)
//  confidence  = 0.99, 0.975                        (one or more levels)
//  window      = 250                                (returns per forecast)
//  models      = historical, parametric, ...        (see kModelNames)
//  combination = equal | select | exp
//  lookback = 100, eta = 1, burn = 100              (combination, see risk/adaptive.hpp)
//  lambda = 0.94, mc_paths = 20000, seed = 42       (EWMA decay, Monte Carlo paths and seed)
//  output      = directory for the report
//The number of threads is not part of the config: it must not change the results, and the tests check it doesn't
namespace qre::engine{
    inline const std::vector<std::pair<std::string, risk::BacktestMethod>>& model_table(){
        static const std::vector<std::pair<std::string, risk::BacktestMethod>> t{
            {"historical", risk::BacktestMethod::Historical}, {"parametric", risk::BacktestMethod::Parametric},
            {"monte_carlo", risk::BacktestMethod::MonteCarlo}, {"ewma", risk::BacktestMethod::EwmaParametric},
            {"fhs", risk::BacktestMethod::FilteredHistorical}, {"garch", risk::BacktestMethod::GarchParametric},
            {"fhs_garch", risk::BacktestMethod::GarchFiltered}, {"garch_t", risk::BacktestMethod::GarchStudentT},
            {"garch_evt", risk::BacktestMethod::GarchEvt}};
        return t;
    }

    inline risk::BacktestMethod model_from_name(const std::string& name){
        for(const auto& [n, m] : model_table()) if(n == name) return m;
        throw InvalidInputError("unknown model '" + name + "'");
    }

    inline std::string model_name(risk::BacktestMethod m){
        for(const auto& [n, mm] : model_table()) if(mm == m) return n;
        return "unknown";
    }

    inline std::string combination_name(risk::Combination c){
        return c == risk::Combination::Equal ? "equal" : (c == risk::Combination::SelectBest ? "select" : "exp");
    }

    struct RunConfig{
        std::string data;
        bool equal_weights = true;
        std::vector<std::pair<std::string, double>> weights;     //when not equal
        std::vector<double> confidence{0.99};
        long window = 250;
        std::vector<std::string> models{"historical", "parametric", "ewma", "fhs", "garch", "fhs_garch", "garch_t", "garch_evt"};
        risk::Combination combination = risk::Combination::Equal;
        long lookback = 100;
        double eta = 1.0;
        long burn = 100;
        double lambda = 0.94;
        long mc_paths = 20000;
        long seed = 42;
        std::string output = "runs/latest";
    };

    namespace config_detail{
        inline std::string trim(const std::string& s){
            const auto b = s.find_first_not_of(" \t\r");
            if(b == std::string::npos) return "";
            const auto e = s.find_last_not_of(" \t\r");
            return s.substr(b, e - b + 1);
        }
        inline std::vector<std::string> split_list(const std::string& s){
            std::vector<std::string> out;
            std::stringstream ss(s);
            std::string item;
            while(std::getline(ss, item, ',')){
                item = trim(item);
                if(item.empty()) throw InvalidInputError("empty item in list '" + s + "'");
                out.push_back(item);
            }
            return out;
        }
        inline double to_double(const std::string& key, const std::string& v){
            std::size_t pos = 0;
            double x = 0.0;
            try{ x = std::stod(v, &pos); } catch(const std::exception&){ pos = 0; }
            if(pos != v.size() || v.empty() || !std::isfinite(x)) throw InvalidInputError("'" + key + "' needs a number, got '" + v + "'");
            return x;
        }
        inline long to_long(const std::string& key, const std::string& v){
            const double x = to_double(key, v);
            if(x != std::floor(x) || std::abs(x) > 1e15) throw InvalidInputError("'" + key + "' needs an integer, got '" + v + "'");
            return static_cast<long>(x);
        }
    }

    //Parse config text; errors carry the line number
    inline RunConfig parse_config(const std::string& text){
        using namespace config_detail;
        RunConfig c;
        std::map<std::string, int> seen;
        std::stringstream in(text);
        std::string line;
        int lineno = 0;
        while(std::getline(in, line)){
            ++lineno;
            const auto hash = line.find('#');
            if(hash != std::string::npos) line = line.substr(0, hash);
            line = trim(line);
            if(line.empty()) continue;
            const auto eq = line.find('=');
            if(eq == std::string::npos) throw InvalidInputError("line " + std::to_string(lineno) + ": expected key = value");
            const std::string key = trim(line.substr(0, eq)), v = trim(line.substr(eq + 1));
            if(v.empty()) throw InvalidInputError("line " + std::to_string(lineno) + ": '" + key + "' has no value");
            if(seen[key]++) throw InvalidInputError("line " + std::to_string(lineno) + ": '" + key + "' given twice");
            try{
                if(key == "data") c.data = v;
                else if(key == "output") c.output = v;
                else if(key == "weights"){
                    if(v == "equal"){ c.equal_weights = true; c.weights.clear(); }
                    else{
                        c.equal_weights = false;
                        for(const std::string& item : split_list(v)){
                            const auto e2 = item.find('=');
                            if(e2 == std::string::npos) throw InvalidInputError("weights must be 'equal' or TICKER=w pairs");
                            c.weights.emplace_back(trim(item.substr(0, e2)), to_double(key, trim(item.substr(e2 + 1))));
                        }
                    }
                }
                else if(key == "confidence"){
                    c.confidence.clear();
                    for(const std::string& item : split_list(v)){
                        const double x = to_double(key, item);
                        if(!(x > 0.5 && x < 1.0)) throw InvalidInputError("confidence must be in (0.5, 1)");
                        c.confidence.push_back(x);
                    }
                }
                else if(key == "window") c.window = to_long(key, v);
                else if(key == "models"){
                    c.models = split_list(v);
                    for(const auto& m : c.models) model_from_name(m);
                }
                else if(key == "combination"){
                    if(v == "equal") c.combination = risk::Combination::Equal;
                    else if(v == "select") c.combination = risk::Combination::SelectBest;
                    else if(v == "exp") c.combination = risk::Combination::ExpWeights;
                    else throw InvalidInputError("combination must be equal, select or exp");
                }
                else if(key == "lookback") c.lookback = to_long(key, v);
                else if(key == "eta") c.eta = to_double(key, v);
                else if(key == "burn") c.burn = to_long(key, v);
                else if(key == "lambda") c.lambda = to_double(key, v);
                else if(key == "mc_paths") c.mc_paths = to_long(key, v);
                else if(key == "seed") c.seed = to_long(key, v);
                else throw InvalidInputError("unknown key '" + key + "'");
            } catch(const InvalidInputError& e){
                const std::string msg = e.what();
                throw InvalidInputError(msg.rfind("line ", 0) == 0 ? msg : "line " + std::to_string(lineno) + ": " + msg);
            }
        }
        if(c.data.empty()) throw InvalidInputError("config needs 'data'");
        if(c.window < 2 || c.lookback < 1 || c.burn < 1 || c.eta < 0.0 || c.mc_paths < 100 || c.seed < 0 ||
           !(c.lambda > 0.0 && c.lambda < 1.0)){
            throw InvalidInputError("need window >= 2, lookback >= 1, burn >= 1, eta >= 0, mc_paths >= 100, seed >= 0, 0 < lambda < 1");
        }
        if(c.models.empty()) throw InvalidInputError("need at least one model");
        for(std::size_t i = 0; i < c.models.size(); ++i){
            for(std::size_t j = i + 1; j < c.models.size(); ++j){
                if(c.models[i] == c.models[j]) throw InvalidInputError("model '" + c.models[i] + "' listed twice");
            }
        }
        return c;
    }

    inline RunConfig load_config(const std::string& path){
        std::ifstream f(path);
        if(!f.is_open()) throw InvalidInputError("cannot open config '" + path + "'");
        std::stringstream ss;
        ss << f.rdbuf();
        return parse_config(ss.str());
    }

    //Canonical text of a config (every key, fixed order) -> recorded in the run manifest
    inline std::string canonical(const RunConfig& c){
        std::ostringstream o;
        o.precision(15);
        o << "data = " << c.data << "\nweights = ";
        if(c.equal_weights) o << "equal";
        else for(std::size_t i = 0; i < c.weights.size(); ++i) o << (i ? ", " : "") << c.weights[i].first << "=" << c.weights[i].second;
        o << "\nconfidence = ";
        for(std::size_t i = 0; i < c.confidence.size(); ++i) o << (i ? ", " : "") << c.confidence[i];
        o << "\nwindow = " << c.window << "\nmodels = ";
        for(std::size_t i = 0; i < c.models.size(); ++i) o << (i ? ", " : "") << c.models[i];
        o << "\ncombination = " << combination_name(c.combination) << "\nlookback = " << c.lookback << "\neta = " << c.eta
          << "\nburn = " << c.burn << "\nlambda = " << c.lambda << "\nmc_paths = " << c.mc_paths << "\nseed = " << c.seed
          << "\noutput = " << c.output << "\n";
        return o.str();
    }
}
