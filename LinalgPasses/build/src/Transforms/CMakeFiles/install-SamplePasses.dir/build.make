# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/anirudh/Desktop/MLIR/LinalgPasses_two/LinalgPasses

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/anirudh/Desktop/MLIR/LinalgPasses_two/LinalgPasses/build

# Utility rule file for install-SamplePasses.

# Include any custom commands dependencies for this target.
include src/Transforms/CMakeFiles/install-SamplePasses.dir/compiler_depend.make

# Include the progress variables for this target.
include src/Transforms/CMakeFiles/install-SamplePasses.dir/progress.make

src/Transforms/CMakeFiles/install-SamplePasses:
	cd /Users/anirudh/Desktop/MLIR/LinalgPasses_two/LinalgPasses/build/src/Transforms && /Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/cmake/data/bin/cmake -DCMAKE_INSTALL_COMPONENT="SamplePasses" -P /Users/anirudh/Desktop/MLIR/LinalgPasses_two/LinalgPasses/build/cmake_install.cmake

install-SamplePasses: src/Transforms/CMakeFiles/install-SamplePasses
install-SamplePasses: src/Transforms/CMakeFiles/install-SamplePasses.dir/build.make
.PHONY : install-SamplePasses

# Rule to build all files generated by this target.
src/Transforms/CMakeFiles/install-SamplePasses.dir/build: install-SamplePasses
.PHONY : src/Transforms/CMakeFiles/install-SamplePasses.dir/build

src/Transforms/CMakeFiles/install-SamplePasses.dir/clean:
	cd /Users/anirudh/Desktop/MLIR/LinalgPasses_two/LinalgPasses/build/src/Transforms && $(CMAKE_COMMAND) -P CMakeFiles/install-SamplePasses.dir/cmake_clean.cmake
.PHONY : src/Transforms/CMakeFiles/install-SamplePasses.dir/clean

src/Transforms/CMakeFiles/install-SamplePasses.dir/depend:
	cd /Users/anirudh/Desktop/MLIR/LinalgPasses_two/LinalgPasses/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/anirudh/Desktop/MLIR/LinalgPasses_two/LinalgPasses /Users/anirudh/Desktop/MLIR/LinalgPasses_two/LinalgPasses/src/Transforms /Users/anirudh/Desktop/MLIR/LinalgPasses_two/LinalgPasses/build /Users/anirudh/Desktop/MLIR/LinalgPasses_two/LinalgPasses/build/src/Transforms /Users/anirudh/Desktop/MLIR/LinalgPasses_two/LinalgPasses/build/src/Transforms/CMakeFiles/install-SamplePasses.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/Transforms/CMakeFiles/install-SamplePasses.dir/depend

