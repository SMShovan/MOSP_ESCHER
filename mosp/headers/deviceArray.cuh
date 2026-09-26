#ifndef DEVICE_ARRAY_CUH
#define DEVICE_ARRAY_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

/**
 * @brief Owning device buffer (freed on destruction; not copyable).
 */
template <typename T> class DeviceArray {
public:
  DeviceArray() = default;
  explicit DeviceArray(size_t count) { allocate(count); }
  ~DeviceArray() { cudaFree(data_); }
  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;
  DeviceArray(DeviceArray &&other) noexcept
      : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }
  DeviceArray &operator=(DeviceArray &&other) noexcept {
    if (this != &other) {
      cudaFree(data_);
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  /// (Re)allocate @p count elements (at least one); contents undefined.
  bool allocate(size_t count) {
    cudaFree(data_);
    data_ = nullptr;
    size_ = count;
    return cudaMalloc(&data_, (count > 0 ? count : 1) * sizeof(T)) ==
           cudaSuccess;
  }
  /// Allocate and copy a host vector.
  bool upload(const std::vector<T> &host) {
    return allocate(host.size()) &&
           (host.empty() ||
            cudaMemcpy(data_, host.data(), host.size() * sizeof(T),
                       cudaMemcpyHostToDevice) == cudaSuccess);
  }
  /// Copy the whole buffer into a host vector (resized).
  bool download(std::vector<T> &host) const {
    host.resize(size_);
    return size_ == 0 ||
           cudaMemcpy(host.data(), data_, size_ * sizeof(T),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
  }

  T *data() { return data_; }
  const T *data() const { return data_; }
  size_t size() const { return size_; }

private:
  T *data_ = nullptr;
  size_t size_ = 0;
};

#endif // DEVICE_ARRAY_CUH
