// SPDX-License-Identifier: Apache-2.0
// libra / libra.exe: 標準入出力で USI（布石拡張）を話す。docs/protocol.md、libra-engine/README.md。
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include "engine.h"

int main() {
  std::ios::sync_with_stdio(false);
  libra_engine::Engine eng;
  std::mutex mu;
  std::condition_variable cv;
  std::deque<std::string> inbox;
  bool eof = false;
  // 読み取りスレッド: go の最中でも stop / quit を受け取る
  std::thread reader([&] {
    std::string line;
    while (std::getline(std::cin, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      // stop は go より後に読まれたものだけ効く（go を読んだ時点で消す。go 本体では消さない）
      if (line == "stop" || line == "quit") eng.stop_flag = true;
      else if (line.rfind("go", 0) == 0) eng.stop_flag = false;
      std::lock_guard<std::mutex> lk(mu);
      inbox.push_back(line);
      cv.notify_one();
    }
    std::lock_guard<std::mutex> lk(mu);
    eof = true;
    inbox.push_back("quit");
    cv.notify_one();
  });
  while (!eng.quit) {
    std::string line;
    {
      std::unique_lock<std::mutex> lk(mu);
      cv.wait(lk, [&] { return !inbox.empty(); });
      line = inbox.front();
      inbox.pop_front();
    }
    eng.handle(line);
  }
  reader.detach();  // stdin で塞がっていても終了する
  return 0;
}
