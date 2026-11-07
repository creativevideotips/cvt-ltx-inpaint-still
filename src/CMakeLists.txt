cmake_minimum_required(VERSION 3.20)
# The bundle must load on macOS 15, whatever macOS and SDK built it. Must come before project().
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum macOS version")
project(LTXInpaintStillOFX LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)  # OFX 1.4 Support uses throw() specs and auto_ptr
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# build.sh passes -DLTXIS_VERSION=1.00 for cvt1.00. Also the plugin version hosts store.
if(NOT DEFINED LTXIS_VERSION)
  set(LTXIS_VERSION "1.0")
endif()
string(REPLACE "." ";" LTXIS_VERSION_PARTS "${LTXIS_VERSION}")
list(GET LTXIS_VERSION_PARTS 0 LTXIS_VERSION_MAJOR)
list(GET LTXIS_VERSION_PARTS 1 LTXIS_VERSION_MINOR)
# "08" would be an invalid octal literal in C++.
string(REGEX REPLACE "^0+([0-9])" "\\1" LTXIS_VERSION_MINOR "${LTXIS_VERSION_MINOR}")

set(OFX_DIR ${CMAKE_SOURCE_DIR}/third_party/openfx)
set(BUNDLE_NAME LTXInpaintStill)
set(BUNDLE_DIR ${CMAKE_BINARY_DIR}/${BUNDLE_NAME}.ofx.bundle)
set(PLUGIN_ID com.creativevideotips.ltx.inpaintstill)

add_library(ofx_support STATIC
  ${OFX_DIR}/Support/Library/ofxsCore.cpp
  ${OFX_DIR}/Support/Library/ofxsImageEffect.cpp
  ${OFX_DIR}/Support/Library/ofxsInteract.cpp
  ${OFX_DIR}/Support/Library/ofxsLog.cpp
  ${OFX_DIR}/Support/Library/ofxsMultiThread.cpp
  ${OFX_DIR}/Support/Library/ofxsParams.cpp
  ${OFX_DIR}/Support/Library/ofxsProperty.cpp
  ${OFX_DIR}/Support/Library/ofxsPropertyValidation.cpp)
target_include_directories(ofx_support PUBLIC ${OFX_DIR}/include ${OFX_DIR}/Support/include)
target_compile_options(ofx_support PRIVATE -w -fvisibility=hidden)

if(NOT APPLE)
  message(FATAL_ERROR "This plugin builds on macOS (Apple Silicon) only")
endif()
enable_language(OBJCXX)

# OpenEXR, built into the plugin so the node reads the DWAB stills it saves. From source at a pinned
# version, with its own Imath and libdeflate, whatever this Mac has installed (BSD-3-Clause, MIT).
include(FetchContent)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(OPENEXR_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(OPENEXR_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(OPENEXR_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(OPENEXR_INSTALL OFF CACHE BOOL "" FORCE)
set(OPENEXR_INSTALL_TOOLS OFF CACHE BOOL "" FORCE)
set(OPENEXR_INSTALL_PKG_CONFIG OFF CACHE BOOL "" FORCE)
set(OPENEXR_FORCE_INTERNAL_IMATH ON CACHE BOOL "" FORCE)
set(OPENEXR_FORCE_INTERNAL_DEFLATE ON CACHE BOOL "" FORCE)
FetchContent_Declare(openexr
  URL https://github.com/AcademySoftwareFoundation/openexr/archive/refs/tags/v3.3.5.tar.gz
  URL_HASH SHA256=cb0c88710c906c9bfc59027eb147e780d508c7be1a90b43af3ec9e3c2987b70d)
FetchContent_MakeAvailable(openexr)
# OpenEXR needs C++17, which the OFX support library cannot compile under, so its use stays in here.
# No ofx/src on its include path: our half.h would hide Imath's.
add_library(ltxis_plate STATIC ofx/src/plate_io.cpp)
target_link_libraries(ltxis_plate PRIVATE OpenEXR::OpenEXR)
target_compile_options(ltxis_plate PRIVATE -fvisibility=hidden)

add_library(${BUNDLE_NAME} MODULE
  ofx/src/plugin.cpp
  ofx/src/log.cpp
  ofx/src/user_config.cpp
  ofx/src/platform_macos.cpp
  ofx/src/platform_macos_metal.mm
  ofx/src/transport.cpp
  ofx/src/process.cpp
  ofx/src/marker.cpp)
target_link_libraries(${BUNDLE_NAME} PRIVATE ofx_support ltxis_plate "-framework Metal" "-framework Foundation"
  pthread)
target_compile_definitions(${BUNDLE_NAME} PRIVATE LTXIS_VERSION="${LTXIS_VERSION}"
  LTXIS_VERSION_MAJOR=${LTXIS_VERSION_MAJOR} LTXIS_VERSION_MINOR=${LTXIS_VERSION_MINOR})
target_compile_options(${BUNDLE_NAME} PRIVATE -Wall -Wextra -Wno-deprecated-declarations -Wno-deprecated
  -fvisibility=hidden)
target_link_options(${BUNDLE_NAME} PRIVATE -Wl,-exported_symbols_list,${OFX_DIR}/Support/include/osxSymbols)
file(WRITE ${BUNDLE_DIR}/Contents/Info.plist
"<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\">
<dict>
  <key>CFBundleExecutable</key><string>${BUNDLE_NAME}.ofx</string>
  <key>CFBundleIdentifier</key><string>${PLUGIN_ID}</string>
  <key>CFBundleName</key><string>LTX Inpaint Still</string>
  <key>CFBundlePackageType</key><string>BNDL</string>
  <key>CFBundleVersion</key><string>${LTXIS_VERSION}</string>
  <key>CFBundleShortVersionString</key><string>${LTXIS_VERSION}</string>
</dict>
</plist>
")
set_target_properties(${BUNDLE_NAME} PROPERTIES
  PREFIX "" SUFFIX ".ofx"
  LIBRARY_OUTPUT_DIRECTORY ${BUNDLE_DIR}/Contents/MacOS)

# The Python side goes in the bundle's Resources, refreshed on every build.
add_custom_target(bundle_resources ALL
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${BUNDLE_DIR}/Contents/Resources
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE_DIR}/Contents/Resources/worker
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/worker/ltxag_worker
          ${BUNDLE_DIR}/Contents/Resources/worker/ltxag_worker
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/worker/ltxis_worker
          ${BUNDLE_DIR}/Contents/Resources/worker/ltxis_worker
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/worker/ltx_worker.py
          ${CMAKE_SOURCE_DIR}/worker/still_worker.py ${CMAKE_SOURCE_DIR}/worker/still_job.py
          ${CMAKE_SOURCE_DIR}/worker/still_open_folder.py ${CMAKE_SOURCE_DIR}/worker/still_llm.py
          ${BUNDLE_DIR}/Contents/Resources/worker/
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/helper ${BUNDLE_DIR}/Contents/Resources/helper
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/helper/licence.txt ${CMAKE_SOURCE_DIR}/LICENSE
          ${CMAKE_SOURCE_DIR}/NOTICE ${BUNDLE_DIR}/Contents/Resources/
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/ofx/resources/${PLUGIN_ID}.png
          ${BUNDLE_DIR}/Contents/Resources/${PLUGIN_ID}.png
  DEPENDS ${BUNDLE_NAME})

enable_testing()
set(WORKER_TEST_PYTHON /usr/bin/python3 CACHE FILEPATH "A Python with numpy, for the plate reader test")
add_executable(test_plate_io ofx/tests/test_plate_io.cpp)
target_include_directories(test_plate_io PRIVATE ofx/src)
target_link_libraries(test_plate_io PRIVATE ltxis_plate)
add_test(NAME plate_io COMMAND test_plate_io ${WORKER_TEST_PYTHON} ${CMAKE_SOURCE_DIR}/worker)
add_executable(test_load_bundle ofx/tests/test_load_bundle.cpp)
target_include_directories(test_load_bundle PRIVATE ${OFX_DIR}/include)
target_link_libraries(test_load_bundle PRIVATE dl)
add_dependencies(test_load_bundle ${BUNDLE_NAME})
add_test(NAME load_bundle COMMAND test_load_bundle ${BUNDLE_DIR}/Contents/MacOS/${BUNDLE_NAME}.ofx)
