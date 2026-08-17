add_executable(convergence_probability
    01-probability/convergence.cpp
)

target_link_libraries(convergence_probability
    PRIVATE
        qre
)
