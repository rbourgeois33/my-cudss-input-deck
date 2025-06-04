#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <vector>

enum cudssMatrixViewType_t {
    CUDSS_MVIEW_FULL,
    CUDSS_MVIEW_UPPER,
    CUDSS_MVIEW_LOWER
};

#include "matrix_market_reader.h" 

void write_temp_file(const std::string& filename, const std::string& content) {
    std::ofstream out(filename);
    out << content;
    out.close();
}

TEST(MatrixReaderTest, BasicUnsortedMatrix) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "5 5 4\n"
        "3 2 3.2\n"
        "1 1 1.0\n"
        "2 5 2.5\n"
        "5 5 5.5\n";
    write_temp_file("test1.mtx", content);

    int n, nnz, *offsets, *cols;
    double* values;
    int result = matrix_reader<double>("test1.mtx", n, nnz, &offsets, &cols, &values, CUDSS_MVIEW_FULL, false, false);

    ASSERT_EQ(result, 0);
    ASSERT_EQ(n, 5);
    ASSERT_EQ(nnz, 4);

    ASSERT_EQ(offsets[0], 0);
    ASSERT_EQ(offsets[1], 1); 
    ASSERT_EQ(offsets[2], 2);
    ASSERT_EQ(offsets[3], 3);
    ASSERT_EQ(offsets[4], 3);
    ASSERT_EQ(offsets[5], 4); 

    ASSERT_EQ(values[0], 1.0);
    ASSERT_EQ(values[1], 2.5);
    ASSERT_EQ(values[2], 3.2);
    ASSERT_EQ(values[3], 5.5);

    ASSERT_EQ(cols[0], 0);
    ASSERT_EQ(cols[1], 4);
    ASSERT_EQ(cols[2], 1);
    ASSERT_EQ(cols[3], 4);

    free(offsets);
    free(cols);
    free(values);
}

TEST(MatrixReaderTest, InvalidUpperWithLowerEntry) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 2\n"
        "1 2 1.0\n"
        "3 1 2.0\n";  // lower triangle

    write_temp_file("test2.mtx", content);

    int n, nnz, *offsets, *cols;
    double* values;
    int result = matrix_reader<double>("test2.mtx", n, nnz, &offsets, &cols, &values, CUDSS_MVIEW_UPPER, false, false);

    ASSERT_EQ(result, MtxReaderErrorUpperViewButLowerFound); // Should fail

    free(offsets);
    free(cols);
    free(values);
}

TEST(MatrixReaderTest, InvalidLowerWithUpperEntry) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 2\n"
        "2 1 1.0\n"
        "1 3 2.0\n"; // upper triangle

    write_temp_file("test3.mtx", content);

    int n, nnz, *offsets, *cols;
    double* values;
    int result = matrix_reader<double>("test3.mtx", n, nnz, &offsets, &cols, &values, CUDSS_MVIEW_LOWER, false, false);

    ASSERT_EQ(result, MtxReaderErrorLowerViewButUpperFound); // Should fail
    
    free(offsets);
    free(cols);
    free(values);
}


TEST(MatrixReaderTest, EmptyRowsPresent) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "4 4 2\n"
        "1 1 1.0\n"
        "4 4 4.0\n";

    write_temp_file("test5.mtx", content);

    int n, nnz, *offsets, *cols;
    double* values;
    int result = matrix_reader<double>("test5.mtx", n, nnz, &offsets, &cols, &values, CUDSS_MVIEW_FULL, false, false);

    ASSERT_EQ(result, 0);
    ASSERT_EQ(n, 4);
    ASSERT_EQ(nnz, 2);

    ASSERT_EQ(offsets[0], 0);
    ASSERT_EQ(offsets[1], 1); 
    ASSERT_EQ(offsets[2], 1);
    ASSERT_EQ(offsets[3], 1);
    ASSERT_EQ(offsets[4], 2);

    ASSERT_EQ(values[0], 1.0);
    ASSERT_EQ(values[1], 4.0);

    ASSERT_EQ(cols[0], 0);
    ASSERT_EQ(cols[1], 3);


    free(offsets);
    free(cols);
    free(values);
}

TEST(MatrixReaderTest, FileNotFound) {
    int n, nnz, *offsets, *cols;
    double* values;
    int result = matrix_reader<double>("nonexistent_file.mtx", n, nnz, &offsets, &cols, &values, CUDSS_MVIEW_FULL, false, false);

    ASSERT_EQ(result,  MtxReaderErrorFileNotFound); // Should fail

    free(offsets);
    free(cols);
    free(values);
}

TEST(MatrixReaderTest, SortedOutputCSR) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "4 4 5\n"
        "3 2 3.0\n"
        "1 1 1.0\n"
        "4 4 4.0\n"
        "2 3 2.0\n"
        "2 2 1.5\n";  // deliberately out-of-order input

    write_temp_file("test_sorted.mtx", content);

    int n, nnz, *offsets, *cols;
    double* values;
    int result = matrix_reader<double>("test_sorted.mtx", n, nnz, &offsets, &cols, &values, CUDSS_MVIEW_FULL, false, false);

    ASSERT_EQ(result, 0);
    ASSERT_EQ(n, 4);
    ASSERT_EQ(nnz, 5);

    ASSERT_EQ(offsets[0], 0);
    ASSERT_EQ(offsets[1], 1);  // Row 0 has 1 entry
    ASSERT_EQ(offsets[2], 3);  // Row 1 has 2 entries
    ASSERT_EQ(offsets[3], 4);  // Row 2 has 1 entry
    ASSERT_EQ(offsets[4], 5);  // Row 3 has 1 entry

    ASSERT_EQ(cols[0], 0);     // Row 0
    ASSERT_EQ(cols[1], 1);     // Row 1
    ASSERT_EQ(cols[2], 2);     // Row 1
    ASSERT_EQ(cols[3], 1);     // Row 2
    ASSERT_EQ(cols[4], 3);     // Row 3

    ASSERT_DOUBLE_EQ(values[0], 1.0);
    ASSERT_DOUBLE_EQ(values[1], 1.5);
    ASSERT_DOUBLE_EQ(values[2], 2.0);
    ASSERT_DOUBLE_EQ(values[3], 3.0);
    ASSERT_DOUBLE_EQ(values[4], 4.0);

    free(offsets);
    free(cols);
    free(values);
}

TEST(MatrixReaderTest, InvalidRowIndex) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 2\n"
        "-12 1 1.0\n"
        "3 2 2.0\n";  // only lower

    write_temp_file("test6.mtx", content);

    int n, nnz, *offsets, *cols;
    double* values;
    int result = matrix_reader<double>("test6.mtx", n, nnz, &offsets, &cols, &values, CUDSS_MVIEW_FULL, false, false);

    ASSERT_EQ(result, MtxReaderErrorOutOfBoundRowIndex); // Should fail

    free(offsets);
    free(cols);
    free(values);
}

TEST(MatrixReaderTest, InvalidColIndex) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 3\n"
        "1 1 1.0\n"
        "3 -2 2.0\n"
        "2 3 2.0\n";  // only lower

    write_temp_file("test7.mtx", content);

    int n, nnz, *offsets, *cols;
    double* values;
    int result = matrix_reader<double>("test7.mtx", n, nnz, &offsets, &cols, &values, CUDSS_MVIEW_FULL, false, false);

    ASSERT_EQ(result, MtxReaderErrorOfBoundColIndex); // Should fail

    free(offsets);
    free(cols);
    free(values);
}


TEST(MatrixReaderTest, WrongNnz) {
    std::string content =
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 2\n"
        "2 1 1.0\n"
        "3 2 2.0\n"   
        "3 3 2.0\n";  // only lower

    write_temp_file("test8.mtx", content);

    int n, nnz, *offsets, *cols;
    double* values;
    int result = matrix_reader<double>("test8.mtx", n, nnz, &offsets, &cols, &values, CUDSS_MVIEW_FULL, false, false);

    ASSERT_EQ(result, MtxReaderErrorWrongNnz); // Should fail

    free(offsets);
    free(cols);
    free(values);
}