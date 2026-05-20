# CMakeLists.txt для examples/speculative-mlsd/
# Добавляет цель llama-speculative-mlsd в сборку llama.cpp

set(TARGET llama-speculative-mlsd)
add_executable(${TARGET} speculative_mlsd.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
