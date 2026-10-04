#include "src/support/process.h"

#include <subprocess.h>

#include <cerrno>
#include <exception>
#include <stdexcept>
#include <thread>

namespace tepl::support {
namespace {
int join(subprocess_s& process, int* exit_code) {
  int result;
  do {
    errno = 0;
    result = subprocess_join(&process, exit_code);
  } while (result != 0 && errno == EINTR);
  return result;
}

struct Child {
  subprocess_s process{};
  ~Child() {
    if (process.alive) {
      subprocess_terminate(&process);
      join(process, nullptr);
    }
    subprocess_destroy(&process);
  }
};
}  // namespace

ProcessResult runProcess(const std::vector<std::string>& arguments) {
  if (arguments.empty() || arguments.front().empty())
    throw std::invalid_argument("process requires an executable");
  std::vector<const char*> argv;
  for (const auto& argument : arguments) {
    if (argument.find('\0') != std::string::npos)
      throw std::invalid_argument("process argument contains a NUL byte");
    argv.push_back(argument.c_str());
  }
  argv.push_back(nullptr);

  Child child;
  const int options = subprocess_option_search_user_path |
                      subprocess_option_inherit_environment |
                      subprocess_option_combined_stdout_stderr |
                      subprocess_option_enable_async;
  if (subprocess_create(argv.data(), options, &child.process) != 0)
    throw std::runtime_error("cannot start " + arguments.front());

  ProcessResult result{};
  std::exception_ptr read_error;
  // Drain diagnostics concurrently with join, which closes stdin and waits.
  // Waiting first could deadlock a child that fills its output pipe.
  std::thread reader([&] {
    char buffer[4096];
    for (;;) {
      errno = 0;
      const auto size =
          subprocess_read_stdout(&child.process, buffer, sizeof(buffer));
      if (size == 0 && errno == EINTR) continue;
      if (size == 0) break;
      // Keep draining after an allocation failure so the child can still exit.
      if (read_error) continue;
      try {
        result.output.append(buffer, size);
      } catch (...) {
        read_error = std::current_exception();
      }
    }
  });
  const int status = join(child.process, &result.exit_code);
  if (status != 0) subprocess_terminate(&child.process);
  reader.join();
  if (read_error) std::rethrow_exception(read_error);
  if (status != 0)
    throw std::runtime_error("cannot wait for " + arguments.front());
  return result;
}
}  // namespace tepl::support
