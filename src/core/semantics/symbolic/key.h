#pragma once
#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
namespace tepl::core::symbolic {
// Immutable structural identity. Digests accelerate ordering; exact structure
// breaks digest ties, so hash collisions never identify different expressions.
class Key {
  struct Node;
  std::shared_ptr<const Node> node_;

 public:
  Key() = default;
  Key(std::string text);
  Key(const char* text) : Key(std::string(text)) {}
  Key(std::string text, std::vector<Key> children);
  bool empty() const { return !node_; }
  friend class KeyArena;
  friend bool operator==(const Key&, const Key&);
  friend bool operator<(const Key&, const Key&);
  friend Key operator+(const Key&, const Key&);
  Key& operator+=(const Key& other) { return *this = *this + other; }
};
struct Key::Node {
  std::string text;
  std::vector<Key> children;
  std::uint64_t digest = 14695981039346656037ull;
  Node(std::string text, std::vector<Key> children)
      : text(std::move(text)), children(std::move(children)) {
    auto mix = [&](std::uint64_t value) {
      for (int i = 0; i < 8; ++i) {
        digest ^= (value >> (i * 8)) & 255;
        digest *= 1099511628211ull;
      }
    };
    mix(this->text.size());
    for (unsigned char c : this->text) {
      digest ^= c;
      digest *= 1099511628211ull;
    }
    mix(this->children.size());
    for (const auto& child : this->children)
      mix(child.node_ ? child.node_->digest : 0);
  }
};
inline Key::Key(std::string text) : Key(std::move(text), {}) {}
inline Key::Key(std::string text, std::vector<Key> children)
    : node_(
          std::make_shared<const Node>(std::move(text), std::move(children))) {}
inline bool operator==(const Key& a, const Key& b) {
  if (a.node_ == b.node_) return true;
  if (!a.node_ || !b.node_ || a.node_->digest != b.node_->digest) return false;
  std::vector<std::pair<const Key::Node*, const Key::Node*>> pending{
      {a.node_.get(), b.node_.get()}};
  std::set<std::pair<const Key::Node*, const Key::Node*>> seen;
  while (!pending.empty()) {
    auto [left, right] = pending.back();
    pending.pop_back();
    if (left == right) continue;
    if (!left || !right || left->digest != right->digest) return false;
    if (left->text != right->text ||
        left->children.size() != right->children.size())
      return false;
    if (!seen.emplace(left, right).second) continue;
    for (std::size_t i = 0; i < left->children.size(); ++i)
      pending.emplace_back(left->children[i].node_.get(),
                           right->children[i].node_.get());
  }
  return true;
}
inline bool operator<(const Key& a, const Key& b) {
  if (a.node_ == b.node_) return false;
  if (!a.node_ || !b.node_) return !a.node_ && bool(b.node_);
  if (a.node_->digest != b.node_->digest)
    return a.node_->digest < b.node_->digest;
  if (a == b) return false;
  if (a.node_->text != b.node_->text) return a.node_->text < b.node_->text;
  return a.node_->children < b.node_->children;
}
inline Key operator+(const Key& a, const Key& b) {
  std::vector<Key> parts;
  auto append = [&](const Key& k) {
    if (k.node_ && k.node_->text == "$concat")
      parts.insert(parts.end(), k.node_->children.begin(),
                   k.node_->children.end());
    else
      parts.push_back(k);
  };
  append(a);
  append(b);
  return Key("$concat", std::move(parts));
}

// Owned by one proof context. Canonicalization accelerates repeated equality
// checks without a process-wide registry or history-dependent identities.
class KeyArena {
  using Signature = std::pair<std::string, std::vector<const Key::Node*>>;
  std::map<Signature, Key> nodes_;
  std::map<const Key::Node*, std::pair<Key, Key>> memo_;

 public:
  Key intern(const Key& key) {
    if (!key.node_) return {};
    if (auto i = memo_.find(key.node_.get()); i != memo_.end())
      return i->second.second;
    std::vector<Key> children;
    Signature signature{key.node_->text, {}};
    for (const auto& child : key.node_->children) {
      children.push_back(intern(child));
      signature.second.push_back(children.back().node_.get());
    }
    auto found = nodes_.find(signature);
    Key canonical;
    if (found == nodes_.end()) {
      canonical = Key(signature.first, std::move(children));
      nodes_.emplace(std::move(signature), canonical);
    } else
      canonical = found->second;
    memo_.emplace(key.node_.get(), std::pair{key, canonical});
    return canonical;
  }
};
}  // namespace tepl::core::symbolic
