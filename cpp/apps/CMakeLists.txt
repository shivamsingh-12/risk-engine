#Production executables (Phase 14)
add_executable(qre_run qre_run.cpp)
target_link_libraries(qre_run PRIVATE qre qre_warnings)
