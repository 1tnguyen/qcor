#include <cassert>

// Defines implementations of QIR Opaque types

// FIXME - Qubit should be a struct that keeps track of idx
// qreg name, array it comes from, and associated accelerator buffer
// Make this a struct now so that we can upgrade the API later
// more easily.
struct Qubit {
  uint64_t id;
  operator int() const { return id; }
  Qubit(uint64_t idVal) : id(idVal) {}
};

using Result = uint64_t;

struct Array {
  // Vector of bytes
  using Storage = std::vector<int8_t>;
  int8_t *getItemPointer(int64_t index) {
    assert(index >= 0);
    assert(static_cast<uint64_t>(index * m_itemSizeInBytes) < m_storage.size());
    return &m_storage.at(index * m_itemSizeInBytes);
  }
  int8_t *operator[](int64_t index) { return getItemPointer(index); }
  // Ctors
  // Default items are pointers.
  Array(int64_t nbItems, int itemSizeInBytes = sizeof(int8_t *))
      : m_itemSizeInBytes(itemSizeInBytes),
        // Initialized to zero
        m_storage(nbItems * itemSizeInBytes, 0) {
    assert(m_itemSizeInBytes > 0);
  };
  int64_t size() { return m_storage.size() / m_itemSizeInBytes; }
  void clear() { m_storage.clear(); }

private:
  // Must be const, i.e. changing the element size is NOT allowed.
  const int m_itemSizeInBytes;
  Storage m_storage;
};

using TupleHeader = int *;
