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
 * ADJPathFile.h
 *
 *  Created on: May 18, 2013
 *      Author: Paolo Galbiati
 */

#ifndef ADJPATHFILE_H_
#define ADJPATHFILE_H_

#include <stdint.h>
#include <cstring>
#include <string>

#include "platform_config.h"

using namespace std;

typedef struct path_point
{
    float lat;
    float lon;
} path_point_t;

typedef struct path_file
{
    uint32_t points;
    path_point_t* coordinates;
    uint32_t points_check;
    uint32_t checksum;
} path_file_t;

class adj_path_file
{
public:
    adj_path_file(const string& file);
    virtual ~adj_path_file();

    bool load_file();
    void dump();
    void reset();
    float total_distance();

    inline const string& get_path_name() const { return file_name; };

    inline uint32_t get_points_number() const { return path.points; };
    inline path_point_t* get_point(uint32_t index) const
    {
        if (likely(index < path.points))
        {
            return &path.coordinates[index];
        }

        return nullptr;
    }

private:
    static const uint32_t PATH_FILE_CHECKSUM = 0x00ED00ED;
    uint8_t* path_data;
    path_file_t path;
    string file_name;

    bool verify_file_size(uint32_t file_size);
    bool verify_file();
};

#endif /* ADJPATHFILE_H_ */
