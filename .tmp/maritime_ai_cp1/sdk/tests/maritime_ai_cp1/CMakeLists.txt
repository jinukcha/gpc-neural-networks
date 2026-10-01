if(MARITIME_BUILD_TESTS)
    add_executable(maritime_ai_cp1_contract_test
        fishing_cp1_contract_test.cpp)
    target_link_libraries(maritime_ai_cp1_contract_test PRIVATE maritime_core)
    target_include_directories(maritime_ai_cp1_contract_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../tests)
    target_compile_features(maritime_ai_cp1_contract_test PRIVATE cxx_std_20)
    target_compile_definitions(maritime_ai_cp1_contract_test PRIVATE
        MARITIME_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../../.."
        MARITIME_FIXTURE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../../fixtures")
    if(MSVC)
        target_compile_options(maritime_ai_cp1_contract_test PRIVATE
            /W4 /WX /permissive-)
    else()
        target_compile_options(maritime_ai_cp1_contract_test PRIVATE
            -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion
            -Wshadow -Werror)
    endif()
    add_test(
        NAME maritime_ai.r0a.cp1_autonomous_fishing_voyage
        COMMAND maritime_ai_cp1_contract_test)
    set_tests_properties(
        maritime_ai.r0a.cp1_autonomous_fishing_voyage
        PROPERTIES TIMEOUT 120)
endif()
