#include <iostream>
#include <string>
#include <fstream>
#include <thread>
#include <deep_tests.hpp>
#include <types/ulong256.hpp>
#include <unistd.h>
#include <sys/wait.h>


int main(int argc, char* argv[]) {

    Logger logger("ulong-log.log", DEBUG);
    char logBuf[128];
    key_t key = ftok("/tmp", 65);
    if(key == -1){
        printf("Key error %d: %s", errno, strerror(errno));
        return EXIT_FAILURE;
    }

    int msgid = msgget(key, 0600 | IPC_CREAT);
    if(msgid == -1){
        snprintf(logBuf, sizeof(logBuf) - 1, "Queue creation error %d: %s", errno, strerror(errno));
        logger.error(logBuf);
        return EXIT_FAILURE;
    }

    logger.info("Queue created successfully! Attemping to create range slices...");

    TestLimits limits = getLimits<avx::ULong256::storedType>();
    auto slices = equalDistribute(limits, std::thread::hardware_concurrency());
    std::vector<int> procIds;

    
    for(int i = 0; i < slices.size(); ++i){
        pid_t procid = fork();
        if(procid < 0){
            logger.error("Fork has failed!");
            return EXIT_FAILURE;
        }
        else if(procid == 0){
            divisionWorker<avx::ULong256, unsigned long long>(msgid, slices[i]);
            return 0;
        }
        else {
            logger.info("Created new child process with pid " + std::to_string(procid));
            logger.debug("Limits for child: startTestVal = " + std::to_string(slices[i].testStartVal) + " testEndVal = " + std::to_string(slices[i].testEndVal));
            procIds.push_back(procid);
        }
    }

    logger.info("Waiting for children to finish...");

    QueueMessage msg;
    int status;
    pid_t ret_pid;
    std::ofstream csvFile("mismatch_ulong.csv", std::ios_base::trunc);
    unsigned long long totalBytes = 0;
    if(csvFile.is_open())
        csvFile << "Type_name;Operator;First_value;Second_value;Expected_value;Actual_value\n";
    while(procIds.size() && totalBytes < 1'073'741'824) {
        for(auto it = procIds.begin(); it != procIds.end(); ++it)
        {
            ret_pid = waitpid(*it, &status, WNOHANG);
            if(ret_pid == *it){
                logger.info("Child process with pid " + std::to_string(*it) + " has finished with " + std::to_string(status));
                procIds.erase(it);
                break;
            }
            else if (ret_pid == -1)
                logger.error("Cannot get state of pid (" + std::to_string(*it) + ")");
        }

        if(msgrcv(msgid, &msg, sizeof(msg), 1, IPC_NOWAIT) == -1) {
            if(errno == ENOMSG){
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }

            snprintf(logBuf, sizeof(logBuf) - 1, "msgrcv %d:%s", errno, strerror(errno));
            logger.error(logBuf);
        } else {
            if(csvFile.is_open()){
                std::string buf = testEntryToCSV(msg.entry) + '\n';
                totalBytes += buf.length();
                csvFile.write(buf.c_str(), buf.length());
            }
            else
                logger.info(testEntryToCSV(msg.entry));
        }
    }

    if(totalBytes < 1'073'741'824){
        while(msgrcv(msgid, &msg, sizeof(msg), 1, IPC_NOWAIT) != -1) {
            if(csvFile.is_open())
                csvFile << testEntryToCSV(msg.entry) << '\n';
            else
                logger.info(testEntryToCSV(msg.entry));
        }
        logger.info("All children have finished...");
    } else {
        logger.warning("CSV exceeded 1 073 741 824 bytes! Finishing...");
    }

    csvFile.close();
    
    if(msgctl(msgid, IPC_RMID, nullptr) == -1){
        snprintf(logBuf, sizeof(logBuf) - 1, "Error %d:%s", errno, strerror(errno));
        logger.error(logBuf);
    }
    
    logger.info("Exiting...");

    return 0;
}