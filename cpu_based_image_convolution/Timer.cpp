/*
 * File Name: Timer.cpp
 * Purpose: Provides timing functionality for measuring the duration of processes.
 *          This file defines the Timer class which includes methods to start
 *          and stop a timer and calculate the elapsed time in milliseconds.
 *
 * Author: Rabia OZKAN
 * GitHub: https://github.com/rabiaozkan
 * Creation Date: 2023-11-11
 * Last Update: 2023-11-25
 * License: Apache License, Version 2.0
 */

#include "Timer.hpp"

std::chrono::steady_clock::time_point Timer::startTimer() {
    return std::chrono::steady_clock::now();
}

double Timer::stopTimer(std::chrono::steady_clock::time_point start) {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed_milliseconds = end - start;
    return elapsed_milliseconds.count();
}