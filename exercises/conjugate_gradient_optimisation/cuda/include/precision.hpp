#pragma once

#define SINGLE_PRECISION

#ifdef SINGLE_PRECISION
using real = float;
#else
using real = double;
#endif
