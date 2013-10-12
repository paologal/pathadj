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
 * adj_user_paths.cc
 *
 *  Created on: May 20, 2013
 *      Author: Paolo Galbiati
 */


#include "platform_config.h"
#include "adj_user_paths.h"

adj_user_paths::adj_user_paths(const shared_ptr<gpu_device> gpu, const string& dir)
        : gpu(gpu),
          user_dir(dir) {
}

void adj_user_paths::load_paths() {
#ifdef __linux__
    DIR *dpdf;
    int ret;
    struct dirent entry;
    struct dirent* result;

    dpdf = opendir(user_dir.c_str());
    if (nullptr != dpdf) {
        for (ret = readdir_r(dpdf, &entry, &result);
                result != nullptr && ret == 0;
                ret = readdir_r(dpdf, &entry, &result)) {
            string filename = user_dir + '/' + entry.d_name;
            if (filename.substr(filename.find_last_of(".") + 1) == "bin") {
                this->load_path(filename);
            }
        }
    }

    closedir(dpdf);
#endif /* __linux__ */

#ifdef _WIN32
    WIN32_FIND_DATA find_data;
    string search_path = user_dir;
    // Throw on a trailing backslash if not included
    if (!search_path.empty() && search_path[search_path.length() - 1] != '\\') {
        search_path += "\\";
    }
    string dir_path = search_path;
    search_path += "*.bin";
    HANDLE hFind = FindFirstFile(search_path.c_str(), &find_data);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            string filename = dir_path + find_data.cFileName;
            this->load_path(filename);
        }
        while (FindNextFile(hFind, &find_data) != 0);

        DWORD dwError = GetLastError();
        if (dwError != ERROR_NO_MORE_FILES) {
            // Not All Files processed, deal with Error
        }

        FindClose(hFind);
    }
#endif /* _WIN32 */

    TRACE_INFO("Found %d paths\n", (int32_t) file_array.size());
}

void adj_user_paths::load_path(const string& filename) {
    shared_ptr<adj_path> path(new adj_path(gpu, filename));
    if (true == path->load_file()) {
        file_array.push_back(path);
    }
}

void adj_user_paths::reset() {
    for (vector<shared_ptr<adj_path>>::iterator it = file_array.begin();
            it != file_array.end(); ++it) {
        (*it)->reset();
    }
}

adj_user_paths::~adj_user_paths() {
    reset();
}

