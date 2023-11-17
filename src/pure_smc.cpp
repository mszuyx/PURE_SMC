// Import libs
#include <ros/ros.h>
#include <ds4_driver/Report.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/Dense>
#include <pcl_ros/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/common/io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int32MultiArray.h>
#include <sound_play/sound_play.h>
#include <dynamic_reconfigure/server.h>
#include <pure_smc/smc_paramConfig.h>

using namespace Eigen;
using namespace std;
#define PI 3.14159265358979323846

class smc{
public:
    smc();

    struct idx_scan {
        double angle;
        double range;
    };

    // Dynamic reconfig parameters
    double hri_range_imu = 25, hri_range_yaw_right = 17, hri_range_yaw_left = 30, hri_range_fss = 70, cmd_fuse = 0.7;
    double spin_thre_upper = 0.0, spin_thre_lower = 0.0, alarm_thre_upper = 0.33, alarm_thre_lower = 0.25;
    double max_speed = 1.0, ball_radius = 0.1143, eta = 5.0, zeta = 1.0, epsilon = 0.96;
    double FB_coef = 0.5, LR_coef = 1.0;
    float alarm_volume = 1.0;
    int hri_mode = 3;
    bool debug = true, logging = true, self_test_enable = false;

private:
    // Declare sub & pub
    ros::NodeHandle node_handle_;
    ros::Subscriber joy_sub_;
    ros::Subscriber laser_sub_;
    ros::Subscriber odom_sub_;
    ros::Subscriber roboRIO_sub_;
    ros::Publisher vel_cmd_pub_;
    ros::Publisher data_log_pub_;
    ros::Timer timer;

    // Debug topics
    ros::Publisher user_cmd_pub_;
    ros::Publisher robot_cmd_pub_;
    ros::Publisher debug_scan_pub_;
    ros::Publisher debug_scan_pub2_;

    sensor_msgs::LaserScan::Ptr comb_scan;

    sound_play::SoundClient sc;
    const char *alarm_addr = "/home/upboard/catkin_smc/src/sound_test/sound/alarm_short.wav";
    const char *alarm_addr2 = "/home/upboard/catkin_smc/src/sound_test/sound/seatbelt_warning.wav";
    
    // Declare ROS params
    int loop_rate;
    int smc_mode;
    double dist_thre_upper, dist_thre_lower, dist_thre_padding;
    double sample_size, robot_radius, dead_zone;
    double self_test_coef;
    
    // Declare global variable
    bool over_write = false, hri_on = false, hri_err = false, hri_zero = false, self_test = false, once = false;
    double px_fb = 0.0, py_fb = 0.0, qx_fb = 0.0, qy_fb = 0.0, qz_fb = 0.0, qw_fb = 1.0, vx_fb = 0.0, vy_fb = 0.0, az_fb = 0.0;
    double force_x_old = 0.0, force_y_old = 0.0;
    double hri_reset = 0.0;
    int self_test_cnt = 0;
    double self_test_time_old = 0.0, self_test_timer = 0.0;
    geometry_msgs::TwistStamped usr_vel_cmd, vel_cmd;

    // Declare functions
    double signal_cleaning(double in, double dz);
    bool signal_checking(double in, double th);
    void find_min(const vector<float>& sample_vec, double& min);
    void find_min_idx(const vector<idx_scan>& sample_vec, double& min, double& angle);
    double trim_pi(double in);
    void eulerToQuaternion(const double theta, const double speed, Quaternionf& q);
    void joy_callback(const ds4_driver::Report::ConstPtr& msg);
    void laser_callback(const sensor_msgs::LaserScan::ConstPtr& msg);
    void odom_callback(const nav_msgs::Odometry::ConstPtr& msg);
    void roborio_callback(const std_msgs::Float32MultiArray::ConstPtr& msg);
    void timerCallback_ ();
};

// smc::smc():node_handle_("~"){
smc::smc(): node_handle_("~"), 
            comb_scan(new sensor_msgs::LaserScan()){
    // Init ROS related
    ROS_INFO("Inititalizing Ground Plane Segmentation Node...");

    ROS_INFO("hri_range_imu setting: %f", hri_range_imu);
    ROS_INFO("hri_range_yaw_right setting: %f", hri_range_yaw_right);
    ROS_INFO("hri_range_yaw_left setting: %f", hri_range_yaw_left);
    ROS_INFO("hri_range_fss setting: %f", hri_range_fss);
    ROS_INFO("hri control mode: %d", hri_mode);
    ROS_INFO("cmd_fuse: %f", cmd_fuse);
    ROS_INFO("Enter debug mode?: %d", debug);
    ROS_INFO("smc spin_thre_upper: %f", spin_thre_upper);
    ROS_INFO("smc alarm_thre_upper: %f", alarm_thre_upper);
    ROS_INFO("smc alarm_thre_lower: %f", alarm_thre_lower);
    ROS_INFO("max_speed for remote control mapping: %f", max_speed);
    ROS_INFO("ball_radius: %f", ball_radius);
    ROS_INFO("eta for PF: %f", eta);
    ROS_INFO("zeta for PF: %f", zeta);
    ROS_INFO("epsilon for PF: %f", epsilon);

    node_handle_.param("loop_rate", loop_rate, 200);
    ROS_INFO("SMC control loop rate: %d", loop_rate);

    node_handle_.param("smc_mode", smc_mode, 0);
    ROS_INFO("SMC control mode: %d", smc_mode);

    node_handle_.param("dead_zone", dead_zone, 0.05);
    ROS_INFO("Dead zone setting: %f", dead_zone);

    node_handle_.param("sample_size", sample_size, 0.6);
    ROS_INFO("SMC scan sample_size: %f", sample_size);

    node_handle_.param("robot_radius", robot_radius, 0.4);
    ROS_INFO("robot radius: %f", robot_radius);

    node_handle_.param("dist_thre_upper", dist_thre_upper, 2.0);
    ROS_INFO("smc dist_thre_upper: %f", dist_thre_upper);

    node_handle_.param("dist_thre_lower", dist_thre_lower, 0.4);
    ROS_INFO("smc dist_thre_lower: %f", dist_thre_lower);

    node_handle_.param("dist_thre_padding", dist_thre_padding, 0.1);
    ROS_INFO("smc dist_thre_padding: %f", dist_thre_padding);

    node_handle_.param("self_test_coef", self_test_coef, 0.25);
    ROS_INFO("smc self_test_coef: %f", self_test_coef);
    
    // Subscribe to topics
    joy_sub_ = node_handle_.subscribe("/raw_report", 1, &smc::joy_callback, this);
    odom_sub_ = node_handle_.subscribe<nav_msgs::Odometry>("/roboRIO/odom", 1, &smc::odom_callback, this); 
    timer = node_handle_.createTimer(ros::Duration(double(1.0/loop_rate)), boost::bind(&smc::timerCallback_,this)); 
    laser_sub_ = node_handle_.subscribe("/comb_scan", 1, &smc::laser_callback, this);

    if(hri_mode>0){
        roboRIO_sub_ = node_handle_.subscribe<std_msgs::Float32MultiArray>("/roboRIO/stateFeedback", 5, &smc::roborio_callback, this);
    }
    
    // Publish Init
    if(debug){
        user_cmd_pub_ = node_handle_.advertise<geometry_msgs::PoseStamped>("/smc/user_cmd", 1);
        robot_cmd_pub_ = node_handle_.advertise<geometry_msgs::PoseStamped>("/smc/robot_cmd", 1);
        debug_scan_pub_ = node_handle_.advertise<sensor_msgs::LaserScan>("/smc/debug_scan", 1);
        debug_scan_pub2_ = node_handle_.advertise<sensor_msgs::LaserScan>("/smc/debug_scan2", 1);
    }
    
    if(logging){
        ROS_INFO("data log published at: /roboRIO/data_log");
        data_log_pub_ = node_handle_.advertise<std_msgs::Int32MultiArray>("/roboRIO/data_log", 1);
    }

    std::string vel_cmd_topic;
    node_handle_.param<std::string>("vel_cmd_topic", vel_cmd_topic, "/roboRIO/vel_cmd");
    ROS_INFO("vel cmd published at: %s", vel_cmd_topic.c_str());
    vel_cmd_pub_ = node_handle_.advertise<geometry_msgs::TwistStamped>(vel_cmd_topic, 1);

    usr_vel_cmd.twist.linear.x = 0;
    usr_vel_cmd.twist.linear.y = 0;
    usr_vel_cmd.twist.angular.z = 0; 
    vel_cmd.twist.linear.x = 0;
    vel_cmd.twist.linear.y = 0;
    vel_cmd.twist.angular.z = 0; 
}

void smc::find_min(const vector<float>& sample_vec, double& min){
    if (sample_vec.size()>0){
        for(size_t i =0; i<sample_vec.size(); i++){
            if(sample_vec[i]<min){
                min = sample_vec[i];
            }
        }
    }
}

void smc::find_min_idx(const vector<idx_scan>& sample_vec, double& min, double& angle){
    if (sample_vec.size()>0){
        for(size_t i =0; i<sample_vec.size(); i++){
            if(sample_vec[i].range<min){
                min = sample_vec[i].range;
                angle = sample_vec[i].angle;
            }
        }
    }
}

double smc::trim_pi(double in){
    double out = in;
    if(in>0){
        if(in>PI){
            out = in - (2*PI);
        }else{
            out = in;}
    }else{
        if(in<-PI){
            out = in + (2*PI);
        }else{
            out = in;}
    }
    return out;
}

void smc::eulerToQuaternion(const double theta, const double speed, Quaternionf& q){
    float yaw = theta;
    float pitch = ((PI*0.5)*(speed/max_speed))-(PI*0.5);
    q = AngleAxisf(yaw, Vector3f::UnitZ())*AngleAxisf(pitch, Vector3f::UnitY());
}

void smc::laser_callback(const sensor_msgs::LaserScan::ConstPtr& msg){
    *comb_scan = *msg;
}

void smc::timerCallback_ (){
    // ROS_INFO("callback =======================");
    vector<float> range_data = comb_scan->ranges;
    size_t range_size = comb_scan->ranges.size();
    float angle_min = comb_scan->angle_min;
    float angle_max = comb_scan->angle_max;
    float angle_increment = comb_scan->angle_increment;

    vector<float> samples_left, samples_right;
    vector<idx_scan> samples_motion;
    double  sample_motion_min = 7.0, sample_right_min = 7.0, sample_left_min = 7.0;
    double  sample_motion_ang = 0;
    float scale_spin = 1.0, scale = 0.0;
    double critical_dis = 7.0, critical_ang = 0.0;
    // Logging variables
    double force_x_ideal = 0.0, force_y_ideal = 0.0;
    double force_x_corrected = 0.0, force_y_corrected = 0.0;
    double force_x_smoothed = 0.0, force_y_smoothed = 0.0;
    
    // Get human command speed and direction
    double actual_speed_norm = sqrt((vx_fb*vx_fb) + (vy_fb*vy_fb))/max_speed;
    double goal_angle = atan2(usr_vel_cmd.twist.linear.y ,usr_vel_cmd.twist.linear.x);
    double usr_cmd_speed = sqrt((usr_vel_cmd.twist.linear.y*usr_vel_cmd.twist.linear.y) + (usr_vel_cmd.twist.linear.x*usr_vel_cmd.twist.linear.x));

    sensor_msgs::LaserScan debug_scan_msg;
    sensor_msgs::LaserScan debug_scan_msg2;
    if(debug && comb_scan->ranges.size()>0){
        debug_scan_msg.header = comb_scan->header;
        debug_scan_msg.angle_min = angle_min;
        debug_scan_msg.angle_max = angle_max;
        debug_scan_msg.angle_increment = angle_increment;
        debug_scan_msg.scan_time = comb_scan->scan_time;
        debug_scan_msg.time_increment = comb_scan->time_increment;
        debug_scan_msg.range_min = comb_scan->range_min;
        debug_scan_msg.range_max = comb_scan->range_max;
        int size = (comb_scan->angle_max - comb_scan->angle_min) / comb_scan->angle_increment + 1;
        bool invalid_range_is_inf = true;
        debug_scan_msg.ranges.resize(size,invalid_range_is_inf ? std::numeric_limits<float>::infinity() : 0.0);
        debug_scan_msg2 = debug_scan_msg;
    }

	if(smc_mode>0 && comb_scan->ranges.size()>0 && over_write == false){
        double force_x = 0.0, force_y = 0.0;
        
        int force_cnt_x = 0, force_cnt_y = 0;
        // Adj dist_thre between 50% to 150% according to normalized odom speed
        double adj_dist_thre_upper = dist_thre_upper*(actual_speed_norm+0.5); 
        
        for(size_t i=0; i < range_size; i++){
            // double range_curr = max((range_data[i]-robot_radius), 0.000000000001);
            double range_curr = range_data[i];
            double angle_curr = angle_min+(angle_increment*i);
            double angle_diff = angle_curr-goal_angle;
            double cos_angle_curr = cos(angle_curr);
            double sin_angle_curr = sin(angle_curr);
            double range_curr_x = cos_angle_curr * range_curr;
            double range_curr_y = sin_angle_curr * range_curr;

            // Pad some margin of safty for the F-B
            if((abs(range_curr_y) <= 0.2) && (cos_angle_curr>= 0.0)){
                range_curr = range_curr-(dist_thre_padding+0.1);
            }else if((abs(range_curr_y) <= 0.2) && (cos_angle_curr< 0.0)){
                range_curr = range_curr-dist_thre_padding;
            }

            // Sample range data in the CW spinning directions FL + BR
            if((((range_curr_x)>0.1) && ((range_curr_x)<0.4) && (sin_angle_curr<0.0)) || (((range_curr_x)>-0.4) && ((range_curr_x)<-0.1) && (sin_angle_curr>0.0))){
                if(range_curr<7){
                    samples_right.push_back(abs(range_curr_y));
                }

            // Sample range data in the CCW spinning directions FR + BL
            }else if((((range_curr_x)>0.1) && ((range_curr_x)<0.4) && (sin_angle_curr>0.0)) || (((range_curr_x)>-0.4) && ((range_curr_x)<-0.1) && (sin_angle_curr<0.0))){
                if(range_curr<7){
                    samples_left.push_back(abs(range_curr_y));
                }
            }

            // If the current point is inside dist_thre, turn on SMC
            if(isnan(range_curr) == false  && range_curr<=adj_dist_thre_upper && (smc_mode==2 || smc_mode==3)){
                // check if pt belows to F-B and pt in the direction of motion
                if(abs(range_curr_y) <= 0.15 && (cos_angle_curr*usr_vel_cmd.twist.linear.x>0)){
                    // Calculate the potential field magnitude for x axis
                    double mag_x = -FB_coef*eta*((1.0/range_curr)-(1.0/adj_dist_thre_upper))*((1.0/range_curr)-(1.0/adj_dist_thre_upper));
                    // Collect the terms for x axis only when the forces are repulsive
                    force_x += cos_angle_curr * mag_x;
                    force_cnt_x += 1;
                    if(debug){debug_scan_msg.ranges[i] = range_curr;}
                }
                // check if pt belows to L-R and pt in the direction of motion
                if(abs(range_curr_x) <= 0.5 && (sin_angle_curr*usr_vel_cmd.twist.linear.y>0)){
                    // Calculate the potential field magnitude for y axis
                    double mag_y = -LR_coef*eta*((1.0/range_curr)-(1.0/adj_dist_thre_upper))*((1.0/range_curr)-(1.0/adj_dist_thre_upper));
                    // Collect the terms for y axis only when the forces are repulsive
                    force_y += sin_angle_curr * mag_y;
                    force_cnt_y += 1;
                    if(debug){debug_scan_msg2.ranges[i] = range_curr;}
                }
            }

            if(logging){
                // Find most critical angle and distance
                if(range_curr<critical_dis){
                    critical_dis = range_curr;
                    critical_ang = angle_curr;
                }

                // Sample range data in the direction of command
                if (range_curr<7 && ((cos(angle_diff)*range_curr) > 0) && (abs((sin(angle_diff)*range_curr)) <= (sample_size/2)) && (usr_cmd_speed>0.05*max_speed)){
                    idx_scan temp_scan;
                    temp_scan.range = range_curr;
                    temp_scan.angle = angle_curr;
                    samples_motion.push_back(temp_scan);
                }
            }
        }

        if(logging){
            // Find most critical angle and distance in the direction of command
            find_min_idx(samples_motion,sample_motion_min,sample_motion_ang);
        }

        if(abs(usr_vel_cmd.twist.angular.z)>(0.05*max_speed)){
            find_min(samples_right,sample_right_min);
            if(sample_right_min<=spin_thre_upper && usr_vel_cmd.twist.angular.z <0){
                scale_spin = (sample_right_min-spin_thre_lower)/(spin_thre_upper-spin_thre_lower);
                scale_spin = min(max(scale_spin, 0.0f), 1.0f);
            }
            find_min(samples_left,sample_left_min);
            if(sample_left_min<=spin_thre_upper && usr_vel_cmd.twist.angular.z >0){
                scale_spin = (sample_left_min-spin_thre_lower)/(spin_thre_upper-spin_thre_lower);
                scale_spin = min(max(scale_spin, 0.0f), 1.0f);
            }
        }

        if(smc_mode==1 || smc_mode==3){
            double adj_alarm_thre_upper = alarm_thre_upper;
            double critical_x = cos(critical_ang) * critical_dis;
            double critical_y = sin(critical_ang) * critical_dis;
            if(abs(critical_y) <= 0.15){
                if(cos(critical_ang)*usr_vel_cmd.twist.linear.x>0){
                    adj_alarm_thre_upper = alarm_thre_upper*((abs(usr_vel_cmd.twist.linear.x/max_speed))+1.2);
                }
            }else if(abs(critical_x) <= 0.5){
                if(sin(critical_ang)*usr_vel_cmd.twist.linear.y>0){
                    adj_alarm_thre_upper = alarm_thre_upper*((4.0*abs(usr_vel_cmd.twist.linear.y/max_speed))+1.0);
                }
            }
            scale = float((adj_alarm_thre_upper-critical_dis)/(adj_alarm_thre_upper-alarm_thre_lower));
            scale = min(max(scale, 0.0f), alarm_volume);
            
            // Sound the alarm if min dist to obstacle <X 
            if (scale>0.05f){
                sound_play::Sound alarm = sc.waveSound(alarm_addr,scale);
                alarm.repeat();
            }else{
                sc.stopWave(alarm_addr);
            }
        }

        // Publish control output
        vel_cmd.header.stamp = ros::Time::now();
        if(smc_mode==1){
            force_x_ideal = usr_vel_cmd.twist.linear.x;
            force_y_ideal = usr_vel_cmd.twist.linear.y;
            force_x_corrected = usr_vel_cmd.twist.linear.x;
            force_y_corrected = usr_vel_cmd.twist.linear.y;
            force_x_smoothed = usr_vel_cmd.twist.linear.x;
            force_y_smoothed = usr_vel_cmd.twist.linear.y;

            vel_cmd.twist.angular.z = isnan(usr_vel_cmd.twist.angular.z)?0.0:usr_vel_cmd.twist.angular.z;
        }else if(smc_mode==2 || smc_mode==3){
            // Clip the force in x and y axes within max_speed setting
            force_x = min(max(force_x/(force_cnt_x + 1.0), -1.0), 1.0);
            force_y = min(max(force_y/(force_cnt_y + 1.0), -1.0), 1.0);

            // Ideal cmd_vel
            force_x_ideal = usr_vel_cmd.twist.linear.x + (force_x*abs(usr_vel_cmd.twist.linear.x)); 
            force_y_ideal = usr_vel_cmd.twist.linear.y + (force_y*abs(usr_vel_cmd.twist.linear.y));

            // Calculate zeta
            // double zeta_x = (abs(vx_fb)-abs(force_x_ideal)>0)?FB_coef*zeta:0.0;
            // double zeta_y = (abs(vy_fb)-abs(force_y_ideal)>0)?LR_coef*zeta:0.0;
            double zeta_x = FB_coef*zeta;
            double zeta_y = LR_coef*zeta;
            if ((vx_fb*force_x_ideal>0) && (abs(vx_fb)-abs(force_x_ideal)<0)){
                zeta_x = 0.0;
            }
            if ((vy_fb*force_y_ideal>0) && (abs(vy_fb)-abs(force_y_ideal)<0)){
                zeta_y = 0.0;
            }
            
            // Compensated cmd_vel
            force_x_corrected = force_x_ideal-(zeta_x*(vx_fb-force_x_ideal));
            force_y_corrected = force_y_ideal-(zeta_y*(vy_fb-force_y_ideal));

            // Smoothed cmd_vel
            force_x_smoothed = force_x_corrected-(epsilon*(force_x_corrected - force_x_old));
            force_y_smoothed = force_y_corrected-(epsilon*(force_y_corrected - force_y_old));

            force_x_old = force_x_smoothed;
            force_y_old = force_y_smoothed;

            vel_cmd.twist.angular.z = isnan(usr_vel_cmd.twist.angular.z * scale_spin)?0.0:(usr_vel_cmd.twist.angular.z * scale_spin);
        }
        vel_cmd.twist.linear.x = isnan(force_x_smoothed)?0.0:force_x_smoothed;
        vel_cmd.twist.linear.y = isnan(force_y_smoothed)?0.0:force_y_smoothed;
        vel_cmd.twist.linear.z = hri_reset; // sending reset hri signal
        vel_cmd_pub_.publish(vel_cmd);
    } 
    
    else{ // No SMC
        for(size_t i=0; i < range_size; i++){
            double range_curr = range_data[i];
            double angle_curr = angle_min+(angle_increment*i);
            double angle_diff = angle_curr-goal_angle;

            // Pad some margin of safty for the front
            if(abs((sin(angle_curr)*range_curr)) <= 0.2){
                range_curr = range_curr-dist_thre_padding;
            }

            if(logging){
                // Find most critical angle and distance
                if(range_curr<critical_dis){
                    critical_dis = range_curr;
                    critical_ang = angle_curr;
                }

                // Sample range data in the direction of command
                if (range_curr<7 && ((cos(angle_diff)*range_curr) > 0) && (abs((sin(angle_diff)*range_curr)) <= (sample_size/2)) && (usr_cmd_speed>0.05*max_speed)){
                    idx_scan temp_scan;
                    temp_scan.range = range_curr;
                    temp_scan.angle = angle_curr;
                    samples_motion.push_back(temp_scan);
                    if(debug){debug_scan_msg.ranges[i] = range_curr;}
                }
            }
        }

        if(logging){
            // Find most critical angle and distance in the direction of command
            find_min_idx(samples_motion,sample_motion_min,sample_motion_ang);
        }

        vel_cmd.header.stamp = ros::Time::now();
        vel_cmd.twist.linear.x = isnan(usr_vel_cmd.twist.linear.x)?0.0:usr_vel_cmd.twist.linear.x;
        vel_cmd.twist.linear.y = isnan(usr_vel_cmd.twist.linear.y)?0.0:usr_vel_cmd.twist.linear.y;
        vel_cmd.twist.angular.z = isnan(usr_vel_cmd.twist.angular.z)?0.0:usr_vel_cmd.twist.angular.z;
        vel_cmd.twist.linear.z = hri_reset; // sending reset hri signal
        vel_cmd_pub_.publish(vel_cmd);
    }

    // Publish data log
    if(logging){
        int32_t secs = vel_cmd.header.stamp.sec;
        int32_t nsecs = vel_cmd.header.stamp.nsec;

        vector<int32_t> log_vec = {secs, nsecs, 
            int32_t(usr_vel_cmd.twist.linear.x*1000), int32_t(usr_vel_cmd.twist.linear.y*1000), int32_t(usr_vel_cmd.twist.angular.z*1000), 
            int32_t(vel_cmd.twist.linear.x*1000), int32_t(vel_cmd.twist.linear.y*1000), int32_t(vel_cmd.twist.angular.z*1000),
            int32_t(force_x_ideal*1000), int32_t(force_y_ideal*1000),
            int32_t(force_x_corrected*1000), int32_t(force_y_corrected*1000),
            int32_t(force_x_smoothed*1000), int32_t(force_y_smoothed*1000),
            int32_t(px_fb*1000), int32_t(py_fb*1000),
            int32_t(qx_fb*1000), int32_t(qy_fb*1000), int32_t(qz_fb*1000), int32_t(qw_fb*1000),
            int32_t(vx_fb*1000), int32_t(vy_fb*1000), int32_t(az_fb*1000),
            int32_t(critical_dis*1000), int32_t(critical_ang*1000), 
            int32_t(sample_motion_min*1000), int32_t(sample_motion_ang*1000), 
            int32_t(over_write), int32_t(hri_on)};

        std_msgs::Int32MultiArray log_msg;
        // set up dimensions
        log_msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
        log_msg.layout.dim[0].size = log_vec.size();
        log_msg.layout.dim[0].stride = 1;
        log_msg.layout.dim[0].label = "smc_log";
        // copy in the data
        log_msg.data.clear();
        log_msg.data.insert(log_msg.data.end(), log_vec.begin(), log_vec.end());
        data_log_pub_.publish(log_msg);
    }

    if(debug){
        geometry_msgs::PoseStamped user_cmd;
        geometry_msgs::PoseStamped robot_cmd;
        user_cmd.header = comb_scan->header;
        user_cmd.pose.position.x = 0;
        user_cmd.pose.position.y = 0;
        user_cmd.pose.position.z = 0.5;
        robot_cmd.header = comb_scan->header;
        robot_cmd.pose.position.x = 0;
        robot_cmd.pose.position.y = 0;
        robot_cmd.pose.position.z = 0.5;
        double robot_speed = sqrt((vel_cmd.twist.linear.y*vel_cmd.twist.linear.y) + (vel_cmd.twist.linear.x*vel_cmd.twist.linear.x));
        double robot_angle = atan2(vel_cmd.twist.linear.y ,vel_cmd.twist.linear.x);
        double usr_speed = sqrt((usr_vel_cmd.twist.linear.y*usr_vel_cmd.twist.linear.y) + (usr_vel_cmd.twist.linear.x*usr_vel_cmd.twist.linear.x));
        double usr_angle = atan2(usr_vel_cmd.twist.linear.y ,usr_vel_cmd.twist.linear.x); 
        Quaternionf q_user;
        Quaternionf q_robot;
        eulerToQuaternion(usr_angle, usr_speed, q_user);
        eulerToQuaternion(robot_angle, robot_speed, q_robot);
        user_cmd.pose.orientation.w = q_user.w();
        user_cmd.pose.orientation.x = q_user.x();
        user_cmd.pose.orientation.y = q_user.y();
        user_cmd.pose.orientation.z = q_user.z();
        robot_cmd.pose.orientation.w = q_robot.w();
        robot_cmd.pose.orientation.x = q_robot.x();
        robot_cmd.pose.orientation.y = q_robot.y();
        robot_cmd.pose.orientation.z = q_robot.z();
        user_cmd_pub_.publish(user_cmd);
        robot_cmd_pub_.publish(robot_cmd);
        if(debug_scan_msg.ranges.size()>0){
            debug_scan_pub_.publish(debug_scan_msg);
        }
        if(debug_scan_msg2.ranges.size()>0){
            debug_scan_pub2_.publish(debug_scan_msg2);
        }
    }
}

double smc::signal_cleaning(double in, double dz){
    double out;
    if(abs(in)<dz){out = 0.0;}
    else if(in>1.0){out = 1.0;}
    else if(in<-1.0){out = -1.0;}
    else{out = in;}
    return out;
}

bool smc::signal_checking(double in, double th){
    bool check = true;
    if(abs(in)>th){check = false;}
    return check;
}


void smc::roborio_callback(const std_msgs::Float32MultiArray::ConstPtr& msg){ // if main loop rate is 400, avg sub rate is 490~500
//   data[0] = px_fb;
//   data[1] = py_fb;
//   data[2] = p_yaw_fb;
//   data[3] = vx_fb_fb;
//   data[4] = vy_fb_fb;
//   data[5] = v_yaw_fb;
//   data[6] = HRI_MX;
//   data[7] = HRI_MY;
//   data[8] = HRI_pitch;
//   data[9] = HRI_roll;
//   data[10] = HRI_spin;
//   data[11] = chassis_pitch;
//   data[12] = chassis_roll;
//   data[13] = chassis_dpitch.f;
//   data[14] = chassis_droll.f;
//   data[15] = sw.i;

    if(msg->data[15]==11.0){
        hri_on = true;
        sc.say("enabled","voice_rab_diphone",1.0f);
    }else if(msg->data[15]==7.0){
        hri_on = false;
        sc.say("disabled","voice_rab_diphone",1.0f);
    }else if(msg->data[15]==13.0){
        sc.say("calibrated","voice_rab_diphone",1.0f);
    }

    if(signal_checking(msg->data[8], 2*hri_range_imu) == false || signal_checking(msg->data[9], 2*hri_range_imu) == false || 
       signal_checking(msg->data[7], hri_range_fss) == false || signal_checking(msg->data[6], hri_range_fss) == false){
        if(hri_on == true){
            sc.say("Over Range Error","voice_rab_diphone",1.0f);
            hri_on = false;
            hri_err = true;
        }
    }else{
        hri_err = false;
    }

    if(hri_err == true){
        sound_play::Sound alarm2 = sc.waveSound(alarm_addr2,1.0f);
        alarm2.repeat();
    }else{
        sc.stopWave(alarm_addr2);
    }

    if(hri_on == true){
        double lx = 0.0;
        double ly = 0.0;
        double az = 0.0;

        if(hri_mode==1){
            lx = -1*signal_cleaning(msg->data[8]/hri_range_imu,0.0);
            ly = -1*signal_cleaning(msg->data[9]/hri_range_imu,0.0);
            az = signal_cleaning((msg->data[10]>0? msg->data[10]/hri_range_yaw_right:msg->data[10]/hri_range_yaw_left),0.0);
        }else if(hri_mode==2){
            lx = signal_cleaning(msg->data[7]/hri_range_fss,0.0);
            ly = -1*signal_cleaning(msg->data[6]/hri_range_fss,0.0);
            az = signal_cleaning((msg->data[10]>0? msg->data[10]/hri_range_yaw_right:msg->data[10]/hri_range_yaw_left),0.0);
        }else if(hri_mode==3){
            lx = (1.0-cmd_fuse)*(-1*signal_cleaning(msg->data[8]/hri_range_imu,0.0)) + cmd_fuse*(signal_cleaning(msg->data[7]/hri_range_fss,0.0));
            ly = (1.0-cmd_fuse)*(-1*signal_cleaning(msg->data[9]/hri_range_imu,0.0)) + cmd_fuse*(-1*signal_cleaning(msg->data[6]/hri_range_fss,0.0));
            az = signal_cleaning((msg->data[10]>0? msg->data[10]/hri_range_yaw_right:msg->data[10]/hri_range_yaw_left),0.0); 
        }
        usr_vel_cmd.twist.linear.x = max_speed*lx;
        usr_vel_cmd.twist.linear.y = max_speed*ly;
        usr_vel_cmd.twist.angular.z = -max_speed*az;
        hri_zero = false;
    }else if(hri_on == false && hri_zero == false){
        usr_vel_cmd.twist.linear.x = 0.0;
        usr_vel_cmd.twist.linear.y = 0.0;
        usr_vel_cmd.twist.angular.z = 0.0;
        hri_zero = true;
    }
}

void smc::joy_callback(const ds4_driver::Report::ConstPtr& msg){
    // dpad_up
    // dpad_down
    // dpad_left
    // dpad_right
    // button_cross
    // button_circle
    // button_square
    // button_triangle
    hri_reset = 0.0;
    if(msg->button_circle==true){
        hri_on = true;
        sc.say("TES enabled","voice_rab_diphone",1.0f);
    }else if(msg->button_cross==true){
        hri_on = false;
        sc.say("TES disabled","voice_rab_diphone",1.0f);
    }else if(msg->button_triangle==true){
        hri_reset = 100.0;
        sc.say("TES calibrated","voice_rab_diphone",1.0f);
    }

    if(msg->dpad_left==true){
        smc_mode = 0;
        sc.say("No SMC","voice_rab_diphone",1.0f);
    }else if(msg->dpad_down==true){
        smc_mode = 1;
        sc.say("SMC mode one","voice_rab_diphone",1.0f);
    }else if(msg->dpad_right==true){
        smc_mode = 2;
        sc.say("SMC mode two","voice_rab_diphone",1.0f);
    }else if(msg->dpad_up==true){
        smc_mode = 3;
        sc.say("SMC mode three","voice_rab_diphone",1.0f);
    }

    // if(msg->button_l1 ==true || msg->button_r1 ==true){
    //     self_test = false;
    //     self_test_cnt = 0;
    //     self_test_timer = 0.0;
    //     sc.stopWave(alarm_addr2);
    // }

    if(msg->l2_analog >=225 || msg->r2_analog >=225){
        over_write = true;
        self_test = false;
        self_test_cnt = 0;
        self_test_timer = 0.0;
    }else{
        over_write = false;
        once = true;
    }

    if(over_write == true && once == true){
        sc.say("SMC overwrited","voice_rab_diphone",1.0f);
        // alarm.stop();
        sc.stopWave(alarm_addr);
        sc.stopWave(alarm_addr2);
        once = false;
    }

    if(msg->button_square == true && hri_on==false && self_test_enable==true){
        self_test = true;
        self_test_cnt = 0;
        self_test_timer = 0.0;
        sc.say("Auto test activated","voice_rab_diphone",1.0f);
    }

    if(hri_on==false){
        double lx = 0.0;
        double ly = 0.0;
        double az = 0.0;
        if(self_test == true){
            // Swing left and right
            self_test_cnt += 1;
            double vel_mag = 0.5*sin(PI*self_test_cnt/200);
            if(self_test_cnt>4000){
                self_test_cnt = 0;
                vel_mag = 0.0;
            }
            lx = 0.0;
            ly = vel_mag;
            az = 0.0;

            // // Ramp vel forward
            // double self_test_now =  double(msg->header.stamp.sec) + double(msg->header.stamp.nsec)*1e-9;
            // self_test_timer += self_test_now - self_test_time_old;
            // double self_test_vel = self_test_coef*self_test_timer;
            // self_test_vel = min(self_test_vel, 1.0);

            // if (self_test_vel>0.99){
            //     sound_play::Sound alarm2 = sc.waveSound(alarm_addr2,1.0f);
            //     alarm2.repeat();
            // }else{
            //     sc.stopWave(alarm_addr2);
            // }

            // self_test_time_old = self_test_now;
            // lx = self_test_vel;
            // ly = signal_cleaning((127.5-double(msg->left_analog_x))/127.5,dead_zone);
            // az = signal_cleaning((127.5-double(msg->right_analog_x))/127.5,dead_zone);
        }else{
            lx = signal_cleaning((127.5-double(msg->left_analog_y))/127.5,dead_zone);
            ly = signal_cleaning((127.5-double(msg->left_analog_x))/127.5,dead_zone);
            az = signal_cleaning((127.5-double(msg->right_analog_x))/127.5,dead_zone);
        }
        
        usr_vel_cmd.twist.linear.x = max_speed*lx;
        usr_vel_cmd.twist.linear.y = max_speed*ly;
        usr_vel_cmd.twist.angular.z = max_speed*az;
    }

    // std::cout<<"SMC mode: "<< smc_mode <<std::endl;
    // std::cout<<"max_speed: "<< max_speed <<std::endl;
    // std::cout<<"SMC mode: "<< smc_mode <<std::endl;
    // std::cout<<"self_test_enable: "<< self_test_enable <<std::endl;
}

void smc::odom_callback(const nav_msgs::Odometry::ConstPtr& msg){
    // Get position feedback
    px_fb = msg->pose.pose.position.x;
    py_fb = msg->pose.pose.position.y;
    qx_fb = msg->pose.pose.orientation.x;
    qy_fb = msg->pose.pose.orientation.y;
    qz_fb = msg->pose.pose.orientation.z;
    qw_fb = msg->pose.pose.orientation.w;
    // Get velocity feedback
    vx_fb = msg->twist.twist.linear.x/ball_radius; // in rad/s
    vy_fb = msg->twist.twist.linear.y/ball_radius; // in rad/s
    az_fb = msg->twist.twist.angular.z;
}

void drs_callback(pure_smc::smc_paramConfig &config, uint32_t level, smc &my_smc) {
    ROS_INFO("Reconfigure Request: %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %s %s %s %d", 
                config.max_speed,
                config.ball_radius,
                config.hri_range_imu, 
                config.hri_range_yaw_right, 
                config.hri_range_yaw_left, 
                config.hri_range_fss, 
                config.cmd_fuse, 
                config.eta,  
                config.zeta, 
                config.epsilon, 
                config.FB_coef,
                config.LR_coef,
                config.spin_thre_upper,
                config.alarm_thre_upper, 
                config.alarm_thre_lower, 
                config.alarm_volume,
                config.debug?"True":"False",
                config.logging?"True":"False",
                config.self_test_enable?"True":"False",
                config.hri_mode);

    my_smc.max_speed = config.max_speed / config.ball_radius;
    my_smc.ball_radius = config.ball_radius;
    my_smc.hri_range_imu = config.hri_range_imu;
    my_smc.hri_range_yaw_right = config.hri_range_yaw_right;
    my_smc.hri_range_yaw_left = config.hri_range_yaw_left;
    my_smc.hri_range_fss = config.hri_range_fss;
    my_smc.cmd_fuse = config.cmd_fuse;
    my_smc.eta = config.eta;
    my_smc.zeta = config.zeta;
    my_smc.epsilon = config.epsilon;
    my_smc.FB_coef = config.FB_coef,
    my_smc.LR_coef = config.LR_coef,
    my_smc.spin_thre_upper = config.spin_thre_upper;
    my_smc.alarm_thre_upper = config.alarm_thre_upper;
    my_smc.alarm_thre_lower = config.alarm_thre_lower;
    my_smc.alarm_volume = float(config.alarm_volume);
    my_smc.debug = config.debug;
    my_smc.logging = config.logging;
    my_smc.self_test_enable = config.self_test_enable;
    my_smc.hri_mode = config.hri_mode;
}

int main (int argc, char** argv) {
    ros::init(argc, argv, "smc");
    smc mysmc_node;

    dynamic_reconfigure::Server<pure_smc::smc_paramConfig> server;
    dynamic_reconfigure::Server<pure_smc::smc_paramConfig>::CallbackType drs;
    drs = boost::bind(&drs_callback, _1, _2,  boost::ref(mysmc_node));
    server.setCallback(drs);

    // ros::MultiThreadedSpinner spinner(4); // Use 4 threads
    // spinner.spin();
    ros::spin();
    return 0;
 }
