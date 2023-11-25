#ifndef TIMER_HPP
#define TIMER_HPP

/*
 * File Name: Timer.hpp
 *
 * Author: Rabia OZKAN
 * GitHub: https://github.com/rabiaozkan
 * Creation Date: 2023-11-11
 * Last Update: 2023-11-25
 * License: Apache License, Version 2.0
 */

#include <chrono>

class Timer {
public:
    std::chrono::steady_clock::time_point startTimer();
    double stopTimer(std::chrono::steady_clock::time_point start);
};

#endif // TIMER_HPP
