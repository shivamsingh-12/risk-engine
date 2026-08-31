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

add_executable(newton_failure
    04-numerical-methods/newton_failure.cpp
)

target_link_libraries(newton_failure
    PRIVATE
        qre
)