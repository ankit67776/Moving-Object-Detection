#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

void vid_inf(const std::string& vid_path) {
    // Create a VideoCapture object
    cv::VideoCapture cap(vid_path);
    // get the video frames' width and height for proper saving of videos
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    cv::Size frame_size(frame_width, frame_height);
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    std::string output_video = "output_recorded.mp4";

    // create the `VideoWriter()` object
    cv::VideoWriter out(output_video, fourcc, fps, frame_size);

    // Create Background Subtractor MOG2 object
    cv::Ptr<cv::BackgroundSubtractorMOG2> backSub = cv::createBackgroundSubtractorMOG2();

    // Check if camera opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return;
    }
    int count = 0;
    // Read until video is completed
    while (cap.isOpened()) {
        // Capture frame-by-frame
        cv::Mat frame;
        cap >> frame;

        if (!frame.empty()) {
            // Apply background subtraction
            cv::Mat fg_mask;
            backSub->apply(frame, fg_mask);

            // apply global threshold to remove shadows
            cv::Mat mask_thresh;
            cv::threshold(fg_mask, mask_thresh, 180, 255, cv::THRESH_BINARY);

            // set the kernal
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            // Apply erosion
            cv::Mat mask_eroded;
            cv::morphologyEx(mask_thresh, mask_eroded, cv::MORPH_OPEN, kernel);

            // Find contours
            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(mask_eroded, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            int min_contour_area = 500;  // Define your minimum area threshold
            // filtering contours using list comprehension
            std::vector<std::vector<cv::Point>> large_contours;
            for (const auto& cnt : contours) {
                if (cv::contourArea(cnt) > min_contour_area)
                    large_contours.push_back(cnt);
            }

            cv::Mat frame_out = frame.clone();
            for (const auto& cnt : large_contours) {
                cv::Rect rect = cv::boundingRect(cnt);
                cv::rectangle(frame_out, rect, cv::Scalar(0, 0, 200), 3);
            }

            // saving the video file
            out.write(frame_out);

            // Display the resulting frame
            cv::imshow("Frame_final", frame_out);

            // Press Q on keyboard to exit
            if (waitKey(30) == 'q') {
                break;
            }
        }
        else {
            break;
        }
    }

    // When everything done, release the video capture and writer object
    cap.release();
    out.release();
    // Closes all the frames
    cv::destroyAllWindows();
}

int main() {
    std::string input_video = "sample/car.mp4";
    vid_inf(input_video);
    return 0;
}