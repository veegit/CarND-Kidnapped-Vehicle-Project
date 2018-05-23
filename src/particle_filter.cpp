/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

const double NEAR_ZERO = 0.001;
const int NUM_PARTICLES = 100;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  default_random_engine gen;

  num_particles = NUM_PARTICLES;
  particles.resize(num_particles);

  for (int i = 0; i < num_particles; i++) {
    particles[i].x = dist_x(gen);
    particles[i].y  = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.0;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {
    double newXpos, newYpos, newTheta;
    if (fabs(yaw_rate) > NEAR_ZERO){
      newTheta = particles[i].theta + (delta_t * yaw_rate);
      newXpos = particles[i].x + (velocity / yaw_rate) * (sin(newTheta) - sin(particles[i].theta));
      newYpos = particles[i].y + (velocity / yaw_rate) * (cos(particles[i].theta) - cos(newTheta));
    } else {
      newTheta = particles[i].theta;
      newXpos = particles[i].x + (delta_t * velocity * cos(particles[i].theta));
      newYpos = particles[i].y + (delta_t * velocity * sin(particles[i].theta));
    }

    particles[i].x = newXpos + dist_x(gen);
    particles[i].y = newYpos + dist_y(gen);
    particles[i].theta = newTheta + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  //initialize observation probability:
  double distance = 1.0;
  double minDistance = 0.0;

  //run over current observation vector:
  for (unsigned int z=0; z< observations.size(); ++z) {
      minDistance = std::numeric_limits<float>::max();
      for (unsigned int i=0; i< predicted.size(); ++i) {
        distance = dist(observations[z].x, observations[z].y, predicted[i].x, predicted[i].y);
        if (distance < minDistance) {
          minDistance = distance;
          observations[z].id = predicted[i].id;
        }
      }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  for (int i = 0; i < num_particles; i++) {
    double currentWeight = 1.0;
    Particle p = particles[i];
    vector<LandmarkObs> reachableLandmarks;
    vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;
    for (int landmark = 0; landmark < landmarks.size();landmark++) {
      double distance = dist(landmarks[landmark].x_f, landmarks[landmark].y_f, p.x, p.y);
      if (distance < sensor_range) {
        reachableLandmarks.push_back(LandmarkObs{landmarks[landmark].id_i, landmarks[landmark].x_f, landmarks[landmark].y_f});
      }
    }
    vector<LandmarkObs> transformedObservations;
    for (int obs = 0; obs < observations.size(); obs++) {
      LandmarkObs observation = observations[obs];
      double transformedObservationX = observation.x * cos(p.theta) - observation.y * sin(p.theta) + p.x;
      double transformedObservationY = observation.x * sin(p.theta) + observation.y * cos(p.theta) + p.y;
      observation.x = transformedObservationX;
      observation.y = transformedObservationY;
      transformedObservations.push_back(observation);
    }

    dataAssociation(reachableLandmarks, transformedObservations);

    for (int obs = 0; obs < transformedObservations.size(); obs++) {
      LandmarkObs observation = transformedObservations[obs];
      Map::single_landmark_s landmark = landmarks.at(observation.id-1);
      double x = pow(observation.x - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2));
      double y = pow(observation.y - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));
      double w = exp(- (x+y)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
      currentWeight *= w;
    }

    p.weight = currentWeight;
    weights.push_back(p.weight);
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

     std::random_device rd;
     std::mt19937 gen(rd());
     discrete_distribution<int> d(weights.begin(), weights.end());

     vector<Particle> newParticles;
     newParticles.resize(num_particles);

     for (int i = 0 ; i < num_particles ; i++){
          Particle particle = particles[d(gen)];
          newParticles[i] = particle;
     }
     particles = newParticles;
     weights.clear();
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
