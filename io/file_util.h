#pragma once

#include <string>
#include <vector>

#include "boost/algorithm/string/predicate.hpp"
#include "boost/filesystem.hpp"
#include "boost/lexical_cast.hpp"

namespace airos {
namespace base {

// custom file operate tools
enum FileType { TYPE_FILE, TYPE_DIR };

// file name compared type
enum FileCompareType {
  FCT_DIGITAL = 0,
  FCT_LEXICOGRAPHICAL = 1,
  FCT_UNKNOWN = 8
};

class FileUtil {
 public:
  FileUtil() {}

  ~FileUtil() {}

  // check whether file and directory exists
  static bool Exists(const std::string &filename);

  // check whether file exists with [suffix] extension in [path]
  static bool Exists(const std::string &path, const std::string &suffix);

  // check file type : directory or file
  static bool GetType(const std::string &filename, FileType *type);

  // remove file , include file and directory
  static bool DeleteFile(const std::string &filename);

  // rename file
  static bool RenameFile(const std::string &old_file,
                         const std::string &new_file);

  // create folder
  static bool CreateDir(const std::string &dir);

  static bool GetFileContent(const std::string &path, std::string *content);
  static bool ReadLines(const std::string &path,
                        std::vector<std::string> *lines);

  static std::string RemoveFileSuffix(std::string filename);

  // TODO(@huanziqiang): this function stay just for compatibility,
  // should be removed later
  static bool GetFileList(const std::string &path, const std::string &suffix,
                          std::vector<std::string> *files);

  static bool GetFileList(const std::string &path,
                          std::vector<std::string> *files);

  static std::string GetAbsolutePath(const std::string &prefix,
                                     const std::string &relative_path);

  // get file name
  // "/home/work/data/1.txt" -> 1
  static void GetFileName(const std::string &file, std::string *name);

  // return -1 when error occurred
  static int NumLines(const std::string &filename);

  // compare two file's name by digital value
  // "/home/work/data/1.txt" < "/home/user/data/10.txt"
  // "1.txt" < "./data/2.txt"
  static bool CompareFileByDigital(const std::string &file_left,
                                   const std::string &file_right);

  // compare two file's name by lexicographical order
  static bool CompareFileByLexicographical(const std::string &file_left,
                                           const std::string &file_right);

  FileUtil(const FileUtil &) = delete;
  FileUtil &operator=(const FileUtil &) = delete;

 private:
  static bool CompareFile(const std::string &file_left,
                          const std::string &file_right, FileCompareType type);
};
}  // namespace base
}  // namespace airos
