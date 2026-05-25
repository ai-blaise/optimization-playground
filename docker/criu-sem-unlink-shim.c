#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <semaphore.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef int (*SemCloseFn)(sem_t *sem);
typedef sem_t *(*SemOpenFn)(const char *name, int oflag, ...);
typedef int (*SemUnlinkFn)(const char *name);
typedef int (*ShmUnlinkFn)(const char *name);
typedef int (*UnlinkAtFn)(int dirfd, const char *pathname, int flags);
typedef int (*UnlinkFn)(const char *pathname);

static int KeepSemaphores(void) {
  const char *keep = getenv("SGLANG_CRIU_KEEP_POSIX_SEMAPHORES");
  return keep != NULL && strcmp(keep, "1") == 0;
}

static int IsSemaphorePath(const char *pathname) {
  if (pathname == NULL) {
    return 0;
  }
  return strncmp(pathname, "/dev/shm/sem.", 13) == 0 ||
         strncmp(pathname, "/run/shm/sem.", 13) == 0 ||
         strncmp(pathname, "/sem.", 5) == 0 ||
         strncmp(pathname, "sem.", 4) == 0;
}

static int SemaphoreNameToPath(const char *name, char *path, size_t path_size) {
  const char *stem;

  if (name == NULL || path == NULL) {
    return 0;
  }

  if (strncmp(name, "/dev/shm/sem.", 13) == 0) {
    stem = name + 13;
  } else if (strncmp(name, "/run/shm/sem.", 13) == 0) {
    stem = name + 13;
  } else if (strncmp(name, "/sem.", 5) == 0) {
    stem = name + 5;
  } else if (strncmp(name, "sem.", 4) == 0) {
    stem = name + 4;
  } else if (name[0] == '/') {
    stem = name + 1;
  } else {
    stem = name;
  }

  if (stem[0] == '\0' || strchr(stem, '/') != NULL) {
    return 0;
  }

  return snprintf(path, path_size, "/dev/shm/sem.%s", stem) > 0;
}

static sem_t *MapSemaphoreFile(int fd, int created, unsigned int value) {
  sem_t *sem;

  if (created && ftruncate(fd, (off_t)sizeof(sem_t)) != 0) {
    return SEM_FAILED;
  }

  sem = (sem_t *)mmap(NULL, sizeof(sem_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (sem == MAP_FAILED) {
    return SEM_FAILED;
  }

  if (created && sem_init(sem, 1, value) != 0) {
    int saved_errno = errno;
    munmap(sem, sizeof(sem_t));
    errno = saved_errno;
    return SEM_FAILED;
  }
  return sem;
}

static sem_t *OpenPreservedSemaphore(
    const char *name, int oflag, mode_t mode, unsigned int value) {
  char path[512];
  int created = 0;
  int fd;
  int saved_errno;
  sem_t *sem;

  if (!SemaphoreNameToPath(name, path, sizeof(path))) {
    errno = EINVAL;
    return SEM_FAILED;
  }

  if ((oflag & O_CREAT) != 0) {
    fd = open(path, O_RDWR | O_CLOEXEC | O_CREAT | O_EXCL, mode);
    if (fd >= 0) {
      created = 1;
    } else if (errno == EEXIST && (oflag & O_EXCL) == 0) {
      fd = open(path, O_RDWR | O_CLOEXEC);
    }
  } else {
    fd = open(path, O_RDWR | O_CLOEXEC);
  }

  if (fd < 0) {
    return SEM_FAILED;
  }

  sem = MapSemaphoreFile(fd, created, value);
  saved_errno = errno;
  close(fd);
  if (sem == SEM_FAILED && created) {
    unlink(path);
  }
  errno = saved_errno;
  return sem;
}

static SemCloseFn RealSemClose(void) {
  static SemCloseFn sem_close_fn = NULL;
  if (sem_close_fn == NULL) {
    sem_close_fn = (SemCloseFn)dlsym(RTLD_NEXT, "sem_close");
  }
  return sem_close_fn;
}

static SemOpenFn RealSemOpen(void) {
  static SemOpenFn sem_open_fn = NULL;
  if (sem_open_fn == NULL) {
    sem_open_fn = (SemOpenFn)dlsym(RTLD_NEXT, "sem_open");
  }
  return sem_open_fn;
}

static SemUnlinkFn RealSemUnlink(void) {
  static SemUnlinkFn sem_unlink_fn = NULL;
  if (sem_unlink_fn == NULL) {
    sem_unlink_fn = (SemUnlinkFn)dlsym(RTLD_NEXT, "sem_unlink");
  }
  return sem_unlink_fn;
}

static ShmUnlinkFn RealShmUnlink(void) {
  static ShmUnlinkFn shm_unlink_fn = NULL;
  if (shm_unlink_fn == NULL) {
    shm_unlink_fn = (ShmUnlinkFn)dlsym(RTLD_NEXT, "shm_unlink");
  }
  return shm_unlink_fn;
}

static UnlinkAtFn RealUnlinkAt(void) {
  static UnlinkAtFn unlinkat_fn = NULL;
  if (unlinkat_fn == NULL) {
    unlinkat_fn = (UnlinkAtFn)dlsym(RTLD_NEXT, "unlinkat");
  }
  return unlinkat_fn;
}

static UnlinkFn RealUnlink(void) {
  static UnlinkFn unlink_fn = NULL;
  if (unlink_fn == NULL) {
    unlink_fn = (UnlinkFn)dlsym(RTLD_NEXT, "unlink");
  }
  return unlink_fn;
}

int sem_close(sem_t *sem) {
  if (KeepSemaphores()) {
    if (munmap(sem, sizeof(sem_t)) == 0) {
      return 0;
    }
    return -1;
  }

  SemCloseFn sem_close_fn = RealSemClose();
  if (sem_close_fn == NULL) {
    errno = ENOSYS;
    return -1;
  }
  return sem_close_fn(sem);
}

sem_t *sem_open(const char *name, int oflag, ...) {
  mode_t mode = 0600;
  unsigned int value = 0;

  if ((oflag & O_CREAT) != 0) {
    va_list args;

    va_start(args, oflag);
    mode = (mode_t)va_arg(args, int);
    value = va_arg(args, unsigned int);
    va_end(args);
  }

  if (KeepSemaphores()) {
    return OpenPreservedSemaphore(name, oflag, mode, value);
  }

  SemOpenFn sem_open_fn = RealSemOpen();
  if (sem_open_fn == NULL) {
    errno = ENOSYS;
    return SEM_FAILED;
  }

  if ((oflag & O_CREAT) != 0) {
    return sem_open_fn(name, oflag, mode, value);
  }
  return sem_open_fn(name, oflag);
}

int sem_unlink(const char *name) {
  if (KeepSemaphores()) {
    return 0;
  }

  SemUnlinkFn sem_unlink_fn = RealSemUnlink();
  if (sem_unlink_fn == NULL) {
    errno = ENOSYS;
    return -1;
  }
  return sem_unlink_fn(name);
}

int shm_unlink(const char *name) {
  if (KeepSemaphores() && IsSemaphorePath(name)) {
    return 0;
  }

  ShmUnlinkFn shm_unlink_fn = RealShmUnlink();
  if (shm_unlink_fn == NULL) {
    errno = ENOSYS;
    return -1;
  }
  return shm_unlink_fn(name);
}

int unlink(const char *pathname) {
  if (KeepSemaphores() && IsSemaphorePath(pathname)) {
    return 0;
  }

  UnlinkFn unlink_fn = RealUnlink();
  if (unlink_fn == NULL) {
    errno = ENOSYS;
    return -1;
  }
  return unlink_fn(pathname);
}

int unlinkat(int dirfd, const char *pathname, int flags) {
  if (KeepSemaphores() && flags == 0 && IsSemaphorePath(pathname)) {
    return 0;
  }

  UnlinkAtFn unlinkat_fn = RealUnlinkAt();
  if (unlinkat_fn == NULL) {
    errno = ENOSYS;
    return -1;
  }
  return unlinkat_fn(dirfd, pathname, flags);
}

int __shm_unlink(const char *name) {
  return shm_unlink(name);
}

int __unlink(const char *pathname) {
  return unlink(pathname);
}

int __unlinkat(int dirfd, const char *pathname, int flags) {
  return unlinkat(dirfd, pathname, flags);
}
