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

#Extra add_executable for new experiment