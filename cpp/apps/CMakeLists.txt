#Production executables (Phase 14)
add_executable(qre_run qre_run.cpp)
target_link_libraries(qre_run PRIVATE qre qre_warnings)

#Streaming risk over WebSocket (Phase 22)
find_package(Threads REQUIRED)
add_executable(qre_stream qre_stream.cpp)
target_link_libraries(qre_stream PRIVATE qre qre_warnings Threads::Threads)
