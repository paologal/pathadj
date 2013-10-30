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
 * adj_user_paths.h
 *
 *  Created on: May 20, 2013
 *      Author: Paolo Galbiati
 */

#ifndef ADJ_USER_PATHS_H_
#define ADJ_USER_PATHS_H_

#include "gpu_device.h"
#include "adj_path.h"

class adj_user_paths {
 public:
    explicit adj_user_paths(const shared_ptr<gpu_device> gpu,
                            const string& dir);
    virtual ~adj_user_paths();

    void load_path(const string& filename);
    void load_paths();
    void reset();

    inline uint32_t get_paths_number() const {
        return static_cast<uint32_t>(file_array.size());
    }

    inline const shared_ptr<adj_path> get_path(uint32_t index) const {
        if (likely(index < file_array.size())) {
            return file_array[index];
        } else {
            return nullptr;
        }
    }

 private:
    const shared_ptr<gpu_device> gpu;
    string user_dir;
    vector<shared_ptr<adj_path>> file_array;
};

#endif /* ADJ_USER_PATHS_H_ */
