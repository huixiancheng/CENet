#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml


if __name__ == '__main__':
    parser = argparse.ArgumentParser("./visualize.py")
    parser.add_argument(
        '--what', '-w',
        type=str,
        required=True,
        help='Dataset to visualize. No Default',
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset to visualize. No Default',
    )
    parser.add_argument(
        '--sequence', '-s',
        type=str,
        default="00",
        required=False,
        help='Sequence to visualize. Defaults to %(default)s',
    )
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        default=None,
        required=False,
        help='Alternate location for labels, to use predictions folder. '
             'Must point to directory containing the predictions in the proper format '
             ' (see readme)'
             'Defaults to %(default)s',
    )
    parser.add_argument(
        '--ignore_semantics', '-i',
        dest='ignore_semantics',
        default=False,
        action='store_true',
        help='Ignore semantics. Visualizes uncolored pointclouds.'
             'Defaults to %(default)s',
    )
    parser.add_argument(
        '--offset',
        type=int,
        default=0,
        required=False,
        help='Sequence to start. Defaults to %(default)s',
    )
    parser.add_argument(
        '--ignore_safety',
        dest='ignore_safety',
        default=False,
        action='store_true',
        help='Normally you want the number of labels and ptcls to be the same,'
             ', but if you are not done inferring this is not the case, so this disables'
             ' that safety.'
             'Defaults to %(default)s',
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("*" * 80)
    print("INTERFACE:")
    print('Dataset Type', FLAGS.what)
    print("Dataset", FLAGS.dataset)
    print("Sequence", FLAGS.sequence)
    print("Predictions", FLAGS.predictions)
    print("ignore_semantics", FLAGS.ignore_semantics)
    print("ignore_safety", FLAGS.ignore_safety)
    print("offset", FLAGS.offset)
    print("*" * 80)

    if FLAGS.what == "kitti":
        from common.laserscan import LaserScan, SemLaserScan
        from common.laserscanvis import LaserScanVis
    elif FLAGS.what == "poss":
        from common.posslaserscan import LaserScan, SemLaserScan
        from common.posslaserscanvis import LaserScanVis
    else:
        raise TypeError("This type dataset doesn't exist (use kitti or poss)! Exiting...")
    # open config file
    try:
        if FLAGS.what == "kitti":
            print("Opening config file of KITTI")
            CFG = yaml.safe_load(open('config/labels/semantic-kitti.yaml', 'r'))
        elif FLAGS.what == "poss":
            print("Opening config file of POSS")
            CFG = yaml.safe_load(open('config/labels/semantic-poss.yaml', 'r'))
        else:
            raise TypeError("This type dataset doesn't exist (use kitti or poss)! Exiting...")

    except Exception as e:
        raise TypeError("Error opening yaml file.")


    # fix sequence name
    FLAGS.sequence = '{0:02d}'.format(int(FLAGS.sequence))

    # does sequence folder exist?
    scan_paths = os.path.join(FLAGS.dataset, "sequences",
                              FLAGS.sequence, "velodyne")

    if os.path.isdir(scan_paths):
        print("Sequence folder exists! Using sequence from %s" % scan_paths)
    else:
        raise TypeError("Sequence folder doesn't exist from %s! Exiting..." % scan_paths)

    # populate the pointclouds
    scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(scan_paths)) for f in fn]
    scan_names.sort()

    if FLAGS.what == "poss":
        tag_paths = os.path.join(FLAGS.dataset, "sequences",
                                 FLAGS.sequence, "tag")
        tag_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(tag_paths)) for f in fn]
        tag_names.sort()

    # does sequence folder exist?
    if not FLAGS.ignore_semantics:
        if FLAGS.predictions is not None:
            label_paths = os.path.join(FLAGS.predictions, "sequences",
                                       FLAGS.sequence, "predictions")
        else:
            label_paths = os.path.join(FLAGS.dataset, "sequences",
                                       FLAGS.sequence, "labels")
        if os.path.isdir(label_paths):
            print("Labels folder exists! Using labels from %s" % label_paths)
        else:
            raise TypeError("Labels folder doesn't exist from %s ! Exiting..." % label_paths)

        # populate the pointclouds
        label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(label_paths)) for f in fn]
        label_names.sort()

        # check that there are same amount of labels and scans
        if not FLAGS.ignore_safety:
            assert (len(label_names) == len(scan_names))

    # create a scan
    if FLAGS.ignore_semantics:
        scan = LaserScan(project=True)  # project all opened scans to spheric proj
    else:
        color_dict = CFG["color_map"]
        scan = SemLaserScan(color_dict, project=True)

    # create a visualizer
    semantics = not FLAGS.ignore_semantics
    if not semantics:
        label_names = None
    if FLAGS.what == "poss":
        vis = LaserScanVis(scan=scan,
                           scan_names=scan_names,
                           tag_names=tag_names,
                           label_names=label_names,
                           offset=FLAGS.offset,
                           semantics=semantics)

    else:
        vis = LaserScanVis(scan=scan,
                       scan_names=scan_names,
                       label_names=label_names,
                       offset=FLAGS.offset,
                       semantics=semantics,
                       instances=False)
    # print instructions
    print("To navigate:")
    print("\tb: back (previous scan)")
    print("\tn: next (next scan)")
    print("\tq: quit (exit program)")

    # run the visualizer
    vis.run()
