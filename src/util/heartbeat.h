/****
DIAMOND protein sequence aligner
Copyright (C) 2012-2026 Benjamin J. Buchfink

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
****/
// SPDX-License-Identifier: GPL-3.0-or-later

#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

struct Heartbeat {

    explicit Heartbeat(std::function<void()> callback)
        : callback_(std::move(callback)), stop_flag_(false) {

        worker_ = std::thread([this]() {
            std::unique_lock<std::mutex> lock(mutex_);
            while (!stop_flag_) {
                auto status = cv_.wait_for(lock, std::chrono::seconds(1));
                if (status == std::cv_status::timeout) {                    
                    lock.unlock();
                    if (callback_) {
                        callback_();
                    }
                    lock.lock();
                }
            }
            });
    }

    ~Heartbeat() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_flag_ = true;
        }
        cv_.notify_one();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    Heartbeat(const Heartbeat&) = delete;
    Heartbeat& operator=(const Heartbeat&) = delete;

private:
    std::function<void()> callback_;
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_flag_;

};