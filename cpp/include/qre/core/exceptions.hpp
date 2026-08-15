#pragma once
#include <stdexcept>
#include <string>

namespace qre{
    class NumericalError : public std::runtime_error{
        public:
        //Inherits base class's constructor so ConvergenceError gets a constructor that takes a std::string message without writing manually
            using std::runtime_error::runtime_error;
    };
    //easy to catch ConvergenceError of actual convergence
    class ConvergenceError : public NumericalError{
        public:
            using NumericalError::NumericalError;
    };
    //Or something numerical was just incorrect
    class InvalidInputError : public NumericalError{
        public:
            using NumericalError::NumericalError;
    };
}