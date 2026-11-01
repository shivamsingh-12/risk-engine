#pragma once
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/core/types.hpp"
//Data hardening: one validated loader for price CSVs instead of a copy in every experiment
//Format: header "Date,T1,T2,...", one row per trading day, ISO dates strictly increasing
//Empty cells -> NaN (missing). Anything malformed throws with the line number -> fail loudly, never guess
namespace qre::data{
    struct PriceTable{
        std::vector<std::string> dates;
        std::vector<std::string> tickers;
        PriceMatrix prices;             //rows = dates, cols = tickers
    };

    namespace detail{
        inline std::vector<std::string> split_csv_line(const std::string& line){
            std::vector<std::string> cells;
            std::stringstream ss(line);
            std::string cell;
            while(std::getline(ss, cell, ',')) cells.push_back(cell);
            if(!line.empty() && line.back() == ',') cells.emplace_back();   //trailing empty cell
            return cells;
        }
        inline std::string line_error(std::size_t lineno, const std::string& what){
            return "line " + std::to_string(lineno) + ": " + what;
        }
    }

    inline PriceTable load_price_csv(const std::string& path){
        std::ifstream in(path);
        if(!in.is_open()) throw InvalidInputError("cannot open " + path);
        std::string line;
        if(!std::getline(in, line)) throw InvalidInputError("empty file");
        if(!line.empty() && line.back() == '\r') line.pop_back();
        const std::vector<std::string> header = detail::split_csv_line(line);
        PriceTable table;
        table.tickers.assign(header.begin() + (header.empty() ? 0 : 1), header.end());
        std::unordered_set<std::string> seen;
        for(const auto& t : table.tickers){
            if(t.empty() || !seen.insert(t).second) throw InvalidInputError("header must list unique, non-empty tickers");
        }
        if(table.tickers.empty()) throw InvalidInputError("header must list unique, non-empty tickers");

        static const std::regex iso_date(R"(^\d{4}-\d{2}-\d{2}$)");
        std::vector<std::vector<double>> rows;
        std::size_t lineno = 1;
        while(std::getline(in, line)){
            ++lineno;
            if(!line.empty() && line.back() == '\r') line.pop_back();
            if(line.empty()) continue;
            const std::vector<std::string> cells = detail::split_csv_line(line);
            if(cells.size() != header.size()){
                throw InvalidInputError(detail::line_error(lineno, "expected " + std::to_string(header.size()) +
                                        " cells, got " + std::to_string(cells.size())));
            }
            if(!std::regex_match(cells[0], iso_date)) throw InvalidInputError(detail::line_error(lineno, "date must be YYYY-MM-DD"));
            if(!table.dates.empty() && cells[0] <= table.dates.back()){
                throw InvalidInputError(detail::line_error(lineno, "dates must be strictly increasing"));
            }
            std::vector<double> row;
            row.reserve(cells.size() - 1);
            for(std::size_t i = 1; i < cells.size(); ++i){
                if(cells[i].empty()){
                    row.push_back(std::numeric_limits<double>::quiet_NaN());
                    continue;
                }
                std::size_t used = 0;
                double v = 0.0;
                try{ v = std::stod(cells[i], &used); }
                catch(const std::exception&){ used = 0; }
                if(used != cells[i].size()) throw InvalidInputError(detail::line_error(lineno, "non-numeric price"));
                row.push_back(v);
            }
            table.dates.push_back(cells[0]);
            rows.push_back(std::move(row));
        }
        if(rows.empty()) throw InvalidInputError("no data rows");
        table.prices.resize(static_cast<Eigen::Index>(rows.size()), static_cast<Eigen::Index>(table.tickers.size()));
        for(std::size_t r = 0; r < rows.size(); ++r)
            for(std::size_t c = 0; c < table.tickers.size(); ++c)
                table.prices(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = rows[r][c];
        return table;
    }
}
