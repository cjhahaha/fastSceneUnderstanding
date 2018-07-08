cmake_minimum_required(VERSION 2.8)
project( label )

include_directories(/data8T/aucid/LuaJIT/include/luajit-2.0)
#set(CMAKE_CXX_FLAGS "-Waggregate-return -Wcast-align -Wmissing-prototypes -Wnested-externs -Wshadow -Wwrite-strings -pedantic ${CMAKE_CXX_FLAGS}")

find_package( OpenCV REQUIRED )

#add_executable( label label.cpp )
add_library( label SHARED label.cpp )

target_link_libraries( label ${OpenCV_LIBS} ${LUA_INC})
