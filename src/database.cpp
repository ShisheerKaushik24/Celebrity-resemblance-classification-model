#include <iostream>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/dnn.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace dlib;

// define the neural network model for face recognition
using net_type = loss_metric<fc_no_bias<128, avg_pool_everything<
    relu<affine<con<20,5,5,2,2,
    relu<affine<con<40,5,5,2,2,
    relu<affine<con<60,5,5,2,2,
    max_pool<3,3,2,2,
    relu<affine<fc<160,
    input<matrix<unsigned char>>
    >>>>>>>>>>>>;

// function to load the face detection and landmark detection models
void load_models(frontal_face_detector& detector, shape_predictor& sp) {
    detector = get_frontal_face_detector();
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
}

// function to extract face descriptors from an image
void extract_face_descriptors(const matrix<rgb_pixel>& img, std::vector<matrix<float, 0, 1>>& face_descriptors, const net_type& net) {
    // detect faces in the image
    frontal_face_detector detector = get_frontal_face_detector();
    std::vector<rectangle> dets = detector(img);

    // get the face landmarks for each detected face
    shape_predictor sp;
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
    std::vector<matrix<rgb_pixel>> faces;
    for (const auto& det : dets) {
        full_object_detection shape = sp(img, det);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
        faces.push_back(move(face_chip));
    }

    // compute the face descriptors for each face
    face_descriptors.clear();
    for (const auto& face : faces) {
        matrix<float, 0, 1> face_descriptor = net(face);
        face_descriptors.push_back(move(face_descriptor));
    }
}

// function to create a database of labeled face descriptors
void create_face_database(const string& database_path, const net_type& net) {
    // create a vector to store the face descriptors and labels
    std::vector<matrix<float, 0, 1>> face_descriptors;
    std::vector<string> face_labels;

    // loop over all images in the database directory and extract the face descriptors
    for (const auto& entry : directory(database_path)) {
        if (!is_directory(entry.path())) {
            matrix<rgb_pixel> img;
            load_image(img, entry.path().string());

            std::vector<matrix<float, 0, 1>> descriptors;
            extract_face_descriptors(img, descriptors, net);

            face_descriptors.insert(face_descriptors.end(), descriptors.begin(), descriptors.end());
            face_labels.insert(face_labels.end(), descriptors.size(), entry.path().stem().string());
        }
    }

    // save the face descriptors and labels to a file
    serialize("face_database.dat") << face_descriptors << face_labels;
}

// function to load the database of labeled face descriptors
void load_face_database(std::vector<matrix<float, 0, 1>>& face_descriptors, std::vector<string>& face_labels) {
    deserialize("face_database.dat") >> face_descriptors >> face_labels;
}

int main
() {
try {
// load the face detection and landmark detection models
frontal_face_detector detector;
shape_predictor sp;
load_models(detector, sp);
    // load the neural network model for face recognition
    net_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

    // create a face database
    create_face_database("database_directory_path", net);

    // load the face database
    std::vector<matrix<float, 0, 1>> face_descriptors;
    std::vector<string> face_labels;
    load_face_database(face_descriptors, face_labels);

    // initialize the video capture object
    cv::VideoCapture cap(0);

    while (true) {
        // get the next frame from the video capture object
        cv::Mat frame;
        cap >> frame;
        cv_image<bgr_pixel> img(frame);

        // detect faces in the frame
        std::vector<rectangle> dets = detector(img);

        // loop over the detected faces
        for (const auto& det : dets) {
            // get the face landmarks
            full_object_detection shape = sp(img, det);

            // extract the face chip
            matrix<rgb_pixel> face_chip;
            extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);

            // compute the face descriptor
            matrix<float, 0, 1> face_descriptor = net(face_chip);

            // compare the face descriptor to the descriptors in the database
            float min_distance = std::numeric_limits<float>::max();
            string min_label;
            for (size_t i = 0; i < face_descriptors.size(); i++) {
                float distance = length(face_descriptor - face_descriptors[i]);
                if (distance < min_distance) {
                    min_distance = distance;
                    min_label = face_labels[i];
                }
            }

            // draw a rectangle around the face and display the label
            cv::rectangle(frame, cv::Point(det.left(), det.top()), cv::Point(det.right(), det.bottom()), cv::Scalar(0, 255, 0));
            cv::putText(frame, min_label, cv::Point(det.left(), det.bottom()), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        }

        // display the frame
        cv::imshow("Face Recognition", frame);

        // check for exit key
        if (cv::waitKey(1) == 27) {
            break;
        }
    }
}
catch (exception& e) {
    cout << e.what() << endl;
}
return 0;
}