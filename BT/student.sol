// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract StudentData {
    struct Student {
        uint256 id;
        string name;
        uint256 age;
        string major;
    }

    // State variable to store students
    mapping(uint256 => Student) private students;
    uint256 public studentCount;

    // Event to emit when a student is added
    event StudentAdded(uint256 id, string name, uint256 age, string major);

    // Function to add a student
    function addStudent(string memory _name, uint256 _age, string memory _major) public {
        studentCount++;
        students[studentCount] = Student(studentCount, _name, _age, _major);
        emit StudentAdded(studentCount, _name, _age, _major);
    }

    // Function to get student data by ID
    function getStudent(uint256 _id) public view returns (uint256, string memory, uint256, string memory) {
        require(_id > 0 && _id <= studentCount, "Student not found");
        Student memory student = students[_id];
        return (student.id, student.name, student.age, student.major);
    }

    // Function to get the total number of students
    function getTotalStudents() public view returns (uint256) {
        return studentCount;
    }
}