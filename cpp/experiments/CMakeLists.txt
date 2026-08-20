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

#Extra add_executable for new experiment