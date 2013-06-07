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
 * adj_path_file.cpp
 *
 *  Created on: May 18, 2013
 *      Author: Paolo Galbiati
 */

#include <cstdio>
#include <sys/types.h>
#include <sys/stat.h>

#include "platform_config.h"
#include "adj_path_file.h"
#include "latlon.h"

adj_path_file::adj_path_file(const string& file):
    path_data(nullptr),
    file_name(file)
{
    memset(&path, 0, sizeof(path));
};

adj_path_file::~adj_path_file()
{
    reset();
}

void adj_path_file::reset()
{
    file_name.empty();

    if (nullptr != path_data)
    {
        delete [] path_data;
        path_data = nullptr;
    }

    memset(&path, 0, sizeof(path));
}

bool adj_path_file::verify_file_size(uint32_t file_size)
{
    uint32_t points = path.points;
    uint32_t check = ((file_size - sizeof(path.points) - sizeof(path.points_check) - sizeof(path.checksum)) >> 3);

    if (points == 0)
    {
       /* Error handling */
       printf("Empty file!\n");
       return false;
    }

    if (points != check)
    {
       /* Error handling */
       printf("Invalid coordinates %d, expected %d\n", check, points);
       return false;
    }

    return true;
}

bool adj_path_file::verify_file()
{
    if (path.points_check != path.points)
    {
        /* Error handling */
        printf("Invalid coordinates checksum %d, expected %d\n", path.points_check, path.points);
        return false;
    }

    if (PATH_FILE_CHECKSUM != path.checksum)
    {
        /* Error handling */
        printf("Invalid file checksum %d, expected %d\n", path.checksum, PATH_FILE_CHECKSUM);
        return false;
    }

    return true;
}

bool adj_path_file::load_file()
{
    struct stat info;
    if (0 != stat(file_name.c_str(), &info))
    {
        printf("Cannot stat %s\n", file_name.c_str());
        reset();
        return false;
    }

    path_data = new uint8_t[info.st_size];
    memset(path_data, 0, info.st_size);
    if (nullptr == path_data)
    {
        /* Error handling */
        printf("Cannot allocate %d bytes\n", (uint32_t)info.st_size);
        reset();
        return false;
    }

    FILE *fp = fopen(file_name.c_str(), "rb");
    if (nullptr == fp)
    {
        /* Error handling */
        printf("Cannot open %s\n", file_name.c_str());
        reset();
        return false;
    }

    /* Try to read a single block of info.st_size bytes */
    size_t blocks_read = fread(path_data, info.st_size, 1, fp);
    fclose(fp);
    if (1 != blocks_read)
    {
        /* Error handling */
        printf("Cannot read %d bytes from %s\n", (uint32_t)info.st_size, file_name.c_str());
        reset();
        return false;
    }

    path.points = *((uint32_t*)path_data);
    if (true != verify_file_size((uint32_t)info.st_size))
    {
        printf("File %s: invalid file size %d\n", file_name.c_str(), (uint32_t)info.st_size);
        reset();
        return false;
    }

    path.coordinates  = (path_point_t*)(path_data + sizeof(path.points));
    path.points_check = *(uint32_t*)((uint8_t*)path.coordinates + path.points * sizeof(path_point_t));
    path.checksum     = *(uint32_t*)((uint8_t*)path.coordinates + path.points * sizeof(path_point_t) + sizeof(path.points_check));

    if (true != verify_file())
    {
        printf("Invalid file %s\n", file_name.c_str());
        reset();
        return false;
    }

    return true;
}

void adj_path_file::dump()
{
    printf("Filename: %s\n", file_name.c_str());
    for (uint32_t i = 0; i < path.points; ++i)
    {
        printf ("Point %d. Latitude %f, Longitude %f\n", i, path.coordinates[i].lat, path.coordinates[i].lon);
    }
}

float adj_path_file::total_distance()
{
	float total = 0.0;
    if (path.points)
    {
    	for (uint32_t i = 0; i < path.points - 1; ++i)
    	{
    		total += latlon::haversine(get_point(i), get_point(i + 1));
    	}
    }

    return total;
}
