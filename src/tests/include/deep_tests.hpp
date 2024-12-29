#ifndef DEEP_TESTS_HPP__
#define DEEP_TESTS_HPP__

#include <string>
#include <vector>
#include <thread>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <test_utils.hpp>
#include <sys/time.h>
#include <sys/ipc.h>
#include <sys/msg.h>


enum LogLevel{DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3, CRITICAL = 4};

class Logger {
    private:
        std::string filename;
        bool logFailed;
        LogLevel logLevel;
        std::string logLvlToStr(LogLevel level){
            switch(level){
                case DEBUG:
                    return "DEBUG";
                case INFO:
                    return "INFO";
                case WARNING:
                    return "WARNING";
                case ERROR:
                    return "ERROR";
                case CRITICAL:
                    return "CRITICAL";
                default:
                    break;
            }
            return "__???__";
        }
    public:
        Logger(std::string filename, const LogLevel logLevel = LogLevel::INFO): 
            logLevel(logLevel),
            filename(filename),
            logFailed(false) {

            std::ofstream logFile(filename, std::ios_base::trunc);
            if(!logFile.is_open())
                throw std::runtime_error("Cannot open \"" + filename + "\"! Log file might be incomplete.");
            else{
                logFile << "Start of log file. Log level set to " << logLvlToStr(logLevel) << std::endl;
                logFile.close();
            }
                
        }

        void setLogLevel(const LogLevel level){
            logLevel = level;
        }

        void log(const LogLevel level, const std::string message) {
            if(level >= logLevel){
                
                time_t now = time(0);
                timeval tmnow;
                tm* timeinfo = localtime(&now);
                gettimeofday(&tmnow, NULL);
                char timestamp[20], tmswusec[28];

                strftime(
                    timestamp, 
                    sizeof(timestamp),
                    "%Y-%m-%d %H:%M:%S", timeinfo
                );

                snprintf(tmswusec, sizeof(tmswusec), "[%s.%03ld] ", timestamp, tmnow.tv_usec / 1000);
                std::string logMessage(tmswusec);
                logMessage += logLvlToStr(level) + ": " + message;
                std::cout << logMessage << std::endl;

                std::ofstream logFile(filename, std::ios_base::app | std::ios_base::out);


                if(logFile.is_open()){
                    logFile << logMessage << std::endl;
                    logFile.close();
                }
                else if(!logFailed){
                    logFailed = true;
                    std::cerr << tmswusec << "LOGERROR: Cannot open \"" << filename << "\"! Logfile will be incomplete.\nPlease check file permissions and restart logging.\nPlease note that this message will only be logged once.\n";
                }
            }
        }

        void info(const std::string message){
            this->log(INFO, message);
        }

        void debug(const std::string message){
            this->log(DEBUG, message);
        }

        void warning(const std::string message){
            this->log(WARNING, message);
        }

        void error(const std::string message){
            this->log(ERROR, message);
        }

        void critical(const std::string message){
            this->log(CRITICAL, message);
        }
};

struct FailedTestEntry {
    char typeName[32];
    char operatorName[4];
    char firstVal[22];
    char secondVal[22];
    char expectedValue[22];
    char actualValue[22];
};

struct QueueMessage {
    long msgType;
    FailedTestEntry entry;
};

template <typename S>
struct TestLimits {
    S minVal;
    S maxVal;
    S testStartVal;
    S testEndVal;
};

std::string testEntryToCSV(const FailedTestEntry& entry, const char separator = ';') {
    char tmp[129];
    tmp[128] = '\0';
    #ifdef _MSC_VER
        static_assert<1 == 2, "This part will not work on Windows!">;
    #else
        snprintf(
            tmp, 
            128, 
            "\"%.32s\"%c\"%.4s\"%c%.22s%c%.22s%c%.22s%c%.22s",
            entry.typeName,
            separator,
            entry.operatorName,
            separator,
            entry.firstVal,
            separator,
            entry.secondVal,
            separator,
            entry.expectedValue,
            separator,
            entry.actualValue
        );
    #endif
    return tmp;
}

template <typename S>
FailedTestEntry createFailedTestEntry(const char* typeName, const char* opName, S expectedVal, S actualVal){
    FailedTestEntry entry;
    snprintf(entry.typeName, sizeof(entry.typeName) - 1,"%s", typeName);
    snprintf(entry.operatorName, sizeof(entry.operatorName) - 1, "%s", opName);
    snprintf(entry.expectedValue, sizeof(entry.expectedValue) - 1, "%s", std::to_string(expectedVal).c_str());
    snprintf(entry.actualValue, sizeof(entry.actualValue) - 1, "%s", std::to_string(actualVal).c_str());
    
    return entry;
}

template <typename S>
FailedTestEntry& updateFailedTestEntry(FailedTestEntry& entry, S fVal, S sVal, S expectedVal, S actualVal){
    snprintf(entry.firstVal, sizeof(entry.firstVal) - 1, "%s", std::to_string(fVal).c_str());
    snprintf(entry.secondVal, sizeof(entry.secondVal) - 1, "%s", std::to_string(sVal).c_str());
    snprintf(entry.expectedValue, sizeof(entry.expectedValue) - 1, "%s", std::to_string(expectedVal).c_str());
    snprintf(entry.actualValue, sizeof(entry.actualValue) - 1, "%s", std::to_string(actualVal).c_str());
    
    return entry;
}

template <typename S>
TestLimits<S> getLimits(){
    S minVal = 0;
    S maxVal = ~minVal; // maxVal is all FFFs
    if(maxVal < minVal) { // If maxVal < 0
        minVal = (static_cast<S>(1) << ((sizeof(S) * 8) - 1)); // Minval is 0x800000... (0b1000000...)
        maxVal ^= minVal; // XOR minVal and maxVal to get actual bounds
    }
    TestLimits<S> limits;
    limits.minVal = minVal;
    limits.maxVal = maxVal;
    limits.testStartVal = minVal;
    limits.testEndVal = maxVal;

    return limits;
}


template<typename T>
std::vector<TestLimits<T>> equalDistribute(const TestLimits<T> &initialLimits, int numOfSlices) {
    unsigned long long distance = initialLimits.maxVal;
    if(initialLimits.minVal) {
        distance += static_cast<unsigned long long>(initialLimits.maxVal);
        ++distance;
    }
    unsigned long long sliceSize = distance / numOfSlices;

    std::vector<TestLimits<T>> results;
    T initialVal = initialLimits.minVal;

    for(int i = 0; i < numOfSlices - 1; ++i) {
        TestLimits<T> sliceLimits;
        sliceLimits.minVal = initialLimits.minVal;
        sliceLimits.maxVal = initialLimits.maxVal,
        sliceLimits.testStartVal = initialVal;
        if(i)
            sliceLimits.testStartVal += 1;
        sliceLimits.testEndVal =  initialVal;

        initialVal += sliceSize;
        sliceLimits.testEndVal = initialVal;
        results.push_back(sliceLimits);
    }

    TestLimits<T> tmpLimits;
    
    tmpLimits.minVal = initialLimits.minVal;
    tmpLimits.maxVal = initialLimits.maxVal;
    tmpLimits.testStartVal = initialVal + 1;
    tmpLimits.testEndVal = initialLimits.maxVal;

    results.push_back(tmpLimits);
    
    return results;
}

bool sendMsg(const int msqid, const QueueMessage& msg){
    if(msgsnd(msqid, &msg, sizeof(msg), 0) == -1) {
        fprintf(stderr, "msgsnd (pid: %d) failed with the following error: %s\n", getpid(), strerror(errno));
        printf("%s\n", testEntryToCSV(msg.entry).c_str());
        return false;
    }

    return true;
}

template <typename S>
bool validateAndSend(QueueMessage& msg, int msqid, S fOp, S sOps[], S expected[], S actual[]){
    char size = 32 / sizeof(S);
    for(char a = 0; a < size; ++a){
        if(expected[a] != actual[a]) {
            updateFailedTestEntry(msg.entry, fOp, sOps[a], expected[a], actual[a]);
            if(!sendMsg(msqid, msg))
                return false;
        }
    }
    return true;
}


template<typename T, typename S = typename T::storedType>
void divisionWorker(const int msqid, const TestLimits<S> &limits) {
    alignas(32) S fstOpBuf[T::size]; // aV
    alignas(32) S scdOpBuf[T::size]; // bV
    alignas(32) S computedBuf[T::size]; // Temporary buffer for comparing values.
    alignas(32) S divVBuf[T::size]; // aV / bV
    alignas(32) S modVBuf[T::size]; // aV % bV

    char counter = 0;
    S i = limits.testStartVal;
    S j;

    auto divEntry = createFailedTestEntry(
        testing::demangle(typeid(T).name()).c_str(), 
        "/",
        0,
        0
    );

    auto modEntry = createFailedTestEntry(
        testing::demangle(typeid(T).name()).c_str(), 
        "%",
        0,
        0
    );

    QueueMessage msg{1, divEntry};
    QueueMessage msgD{1, modEntry};

    do {
        j = limits.testStartVal;
        T aV(i); // Fill aV with i-s.
        counter = 0; // Reset the counter.
        do {
            if(!j){
                ++j; // Skip j = 0;
                continue;
            }

            if(counter == T::size) {
                T bV(scdOpBuf);
                T cV = aV / bV;
                T dV = aV % bV;
                T tV(divVBuf); // Load "valid" values to AVX vector
                T mV(modVBuf); // Load "valid" values to AVX vector
                
                if(cV != tV) { // If not equal then send all that don't match
                    cV.saveAligned(computedBuf); // Load temporary buffer with content of AVX vector.
                    if(!validateAndSend(msg, msqid, i, scdOpBuf, divVBuf, computedBuf))
                        exit(EXIT_FAILURE); // Exit if cannot send message.
                }

                if(dV != mV) {
                    dV.saveAligned(computedBuf);
                    if(!validateAndSend(msgD, msqid, i, scdOpBuf, modVBuf, computedBuf))
                        exit(EXIT_FAILURE);

                }
                counter = -1; // It will be incremented to 0 at the end of loop.
            } else {
                divVBuf[counter] = i / j;
                modVBuf[counter] = i % j;
                scdOpBuf[counter] = j;
            }
            
            ++j;
            ++counter;
        } while(j != limits.testEndVal);
        ++i;
    } while(i != limits.testEndVal);
    
}

#endif