#pragma once
#include <expected>
#include "qre/core/exceptions.hpp"
namespace qre{
    //Defaults error to NumericalError
    //Template holds two values, either a valid value of type T or an error without throwing exception
    // -> caller checks which through returned object
    template <typename T, typename E = NumericalError>
    using Result = std::expected<T,E>;
}