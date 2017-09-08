/*
 * Copyright (C) 2013  Azlos Corporation
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>
 */

/*
 * platform_config.h
 *
 *  Created on: May 31, 2013
 *      Author: Paolo Galbiati
 */

#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <cstring>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <cfloat>

#include <limits>
#include <memory>
#include <thread>
#include <vector>
#include <string>

using std::vector;
using std::string;
using std::shared_ptr;
using std::thread;
using std::numeric_limits;

#ifndef PLATFORM_CONFIG_H_
#define PLATFORM_CONFIG_H_

#ifdef __GNUC__
#if __GNUC_PREREQ(4,7)
#define NUM_CPU thread::hardware_concurrency()
#else
#define NUM_CPU 8
#endif
#endif

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <windows.h>
#undef max
#define likely(x)       (x)
#define unlikely(x)     (x)
#elif __linux__
#include <dirent.h>
#include <unistd.h>

#ifdef __GNUC__
#define likely(x)       __builtin_expect((x), 1)
#define unlikely(x)     __builtin_expect((x), 0)
#else
#define likely(x)       (x)
#define unlikely(x)     (x)

#endif
#endif

#define TRACE_INFO printf
#define TRACE_ERROR printf
#define TRACE_DEBUG printf
#ifdef DEBUG_DUMP
#define TRACE_VERBOSE printf
#else
#define TRACE_VERBOSE
#endif
#endif  /* PLATFORM_CONFIG_H_ */
