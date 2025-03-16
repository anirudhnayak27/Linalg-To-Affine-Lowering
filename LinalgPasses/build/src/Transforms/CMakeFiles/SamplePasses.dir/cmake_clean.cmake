file(REMOVE_RECURSE
  "../../lib/libSamplePasses.a"
  "../../lib/libSamplePasses.pdb"
  "Passes.h.inc"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/SamplePasses.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
