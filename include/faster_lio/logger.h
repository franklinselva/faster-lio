#ifndef FASTER_LIO_LOGGER_H
#define FASTER_LIO_LOGGER_H

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace faster_lio {

inline void InitLogger(spdlog::level::level_enum level = spdlog::level::info) {
    auto logger = spdlog::stdout_color_mt("faster_lio");
    logger->set_level(level);
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_default_logger(logger);
}

}  // namespace faster_lio

#endif  // FASTER_LIO_LOGGER_H
