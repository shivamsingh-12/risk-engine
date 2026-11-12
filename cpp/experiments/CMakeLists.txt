add_executable(convergence_probability
    01-probability/convergence.cpp
)
 
target_link_libraries(convergence_probability
    PRIVATE
        qre
)
 
add_executable(cov_stability
    02-linear-algebra/cov_stability.cpp
)
 
target_link_libraries(cov_stability
    PRIVATE
        qre
)

add_executable(implied_vol_solvers
    04-greeks/implied_vol_solvers.cpp
)

target_link_libraries(implied_vol_solvers
    PRIVATE
        qre
)

add_executable(mc_convergence
    01-mc-convergence/mc_convergence.cpp
)

target_link_libraries(mc_convergence
    PRIVATE
        qre
)

add_executable(mc_vs_bs
    03-bs-vs-mc/mc_vs_bs.cpp
)

target_link_libraries(mc_vs_bs
    PRIVATE
        qre
)

add_executable(var_es_methods
    05-var-es/var_es_methods.cpp
)

target_link_libraries(var_es_methods
    PRIVATE
        qre
)

add_executable(corruption_injection
    data-hardening/corruption_injection.cpp
)

target_link_libraries(corruption_injection
    PRIVATE
        qre
)

add_executable(var_backtest
    backtesting/var_backtest.cpp
)

target_link_libraries(var_backtest
    PRIVATE
        qre
)

add_executable(adaptive_backtest
    adaptive-vol/adaptive_backtest.cpp
)

target_link_libraries(adaptive_backtest
    PRIVATE
        qre
)

add_executable(garch_backtest
    garch/garch_backtest.cpp
)

target_link_libraries(garch_backtest
    PRIVATE
        qre
)

add_executable(garch_t_backtest
    garch-t/garch_t_backtest.cpp
)

target_link_libraries(garch_t_backtest
    PRIVATE
        qre
)

add_executable(stochastic_processes
    stochastic-processes/stochastic_processes.cpp
)

target_link_libraries(stochastic_processes
    PRIVATE
        qre
)

add_executable(heston_experiment
    06-heston/heston_experiment.cpp
)

target_link_libraries(heston_experiment
    PRIVATE
        qre
)

add_executable(heston_calibration
    07-calibration/heston_calibration.cpp
)

target_link_libraries(heston_calibration
    PRIVATE
        qre
)

add_executable(shrinkage_experiment
    08-shrinkage/shrinkage_experiment.cpp
)

target_link_libraries(shrinkage_experiment
    PRIVATE
        qre
)

add_executable(evt_experiment
    09-evt/evt_experiment.cpp
)

target_link_libraries(evt_experiment
    PRIVATE
        qre
)

add_executable(advanced_mc_experiment
    advanced-mc/advanced_mc_experiment.cpp
)

target_link_libraries(advanced_mc_experiment
    PRIVATE
        qre
)

add_executable(adaptive_engine_experiment
    10-adaptive-engine/adaptive_engine_experiment.cpp
)

target_link_libraries(adaptive_engine_experiment
    PRIVATE
        qre
)

add_executable(production_experiment
    production/production_experiment.cpp
)

target_link_libraries(production_experiment
    PRIVATE
        qre
)

#Extra add_executable for new experiment