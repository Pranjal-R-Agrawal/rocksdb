cmake_minimum_required(VERSION 3.10)
project(RocksAdvancedScaling)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Homebrew paths for M1/M2/M3 Macs
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew" "/usr/local")

set(ROCKSDB_DIR "$ENV{HOME}/University/rocksdb")
set(LIBRARY_DIR "$ENV{HOME}/University/rocksdb/build")

add_executable(bench main.cc)

target_compile_definitions(bench PRIVATE OS_MACOSX ROCKSDB_PLATFORM_POSIX GFLAGS=1)
target_include_directories(bench PRIVATE "${ROCKSDB_DIR}/include" "${ROCKSDB_DIR}")
target_link_libraries(bench PRIVATE "${LIBRARY_DIR}/librocksdb.a" "${LIBRARY_DIR}/db_stress_tool/librocksdb_stress.a" gflags)

macro(find_and_link target libname)
    find_library(${libname}_PATH NAMES ${libname})
    if(${libname}_PATH)
        target_link_libraries(${target} PRIVATE ${${libname}_PATH})
    endif()
endmacro()

find_and_link(bench z)
find_and_link(bench bz2)
find_and_link(bench lz4)
find_and_link(bench zstd)
find_and_link(bench snappy)
target_link_libraries(bench PRIVATE pthread dl "-framework Foundation")