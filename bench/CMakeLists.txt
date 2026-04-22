cmake_minimum_required(VERSION 3.10)
project(RocksAdvancedScaling)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Homebrew paths for M1/M2/M3 Macs
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew" "/usr/local")

set(ROCKSDB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(LIBRARY_DIR "${ROCKSDB_DIR}/build")

add_executable(bench main.cc)

if(APPLE)
    target_compile_definitions(bench PRIVATE OS_MACOSX ROCKSDB_PLATFORM_POSIX GFLAGS=1)
else()
    target_compile_definitions(bench PRIVATE OS_LINUX ROCKSDB_PLATFORM_POSIX GFLAGS=1)
endif()

find_path(GFLAGS_INCLUDE_DIR gflags/gflags.h HINTS "$ENV{CMAKE_PREFIX_PATH}/include" "/opt/homebrew/include" "/usr/local/include")
if (GFLAGS_INCLUDE_DIR)
    target_include_directories(bench PRIVATE ${GFLAGS_INCLUDE_DIR})
endif()

find_library(GFLAGS_LIBRARY NAMES gflags HINTS "$ENV{CMAKE_PREFIX_PATH}/lib" "/opt/homebrew/lib" "/usr/local/lib")

target_include_directories(bench PRIVATE "${ROCKSDB_DIR}/include" "${ROCKSDB_DIR}")

if (GFLAGS_LIBRARY)
    target_link_libraries(bench PRIVATE "${LIBRARY_DIR}/db_stress_tool/librocksdb_stress.a" "${LIBRARY_DIR}/librocksdb.a" ${GFLAGS_LIBRARY})
else()
    target_link_libraries(bench PRIVATE "${LIBRARY_DIR}/db_stress_tool/librocksdb_stress.a" "${LIBRARY_DIR}/librocksdb.a" gflags)
endif()

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

if(APPLE)
    target_link_libraries(bench PRIVATE pthread dl "-framework Foundation")
else()
    target_link_libraries(bench PRIVATE pthread dl)
endif()