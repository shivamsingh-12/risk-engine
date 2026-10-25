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

#Extra add_executable for new experiment