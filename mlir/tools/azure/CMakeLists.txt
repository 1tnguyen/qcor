add_executable(azure-qir-tool azure-qir-tool.cpp azure-utils.cpp)
target_include_directories(azure-qir-tool PUBLIC ${XACC_ROOT}/include ${XACC_ROOT}/include/xacc)
target_link_libraries(azure-qir-tool cpr curl)
target_link_directories(azure-qir-tool PRIVATE ${XACC_ROOT}/lib)
target_compile_features(azure-qir-tool
                        PUBLIC
                        cxx_std_17)