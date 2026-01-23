
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def project_points_to_tube(centerline_pts, tube_radius):
    n_pts = len(centerline_pts)

    # Compute tangent vectors using central differences (forward/backward at ends)
    tangents = np.zeros_like(centerline_pts)
    tangents[1:-1] = centerline_pts[2:] - centerline_pts[:-2]
    tangents[0] = centerline_pts[1] - centerline_pts[0]
    tangents[-1] = centerline_pts[-1] - centerline_pts[-2]

    # Normalize tangents
    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangent_norms[tangent_norms == 0] = 1  # Avoid division by zero
    tangents = tangents / tangent_norms

    # Generate random angles around the tube for each point
    angles = np.random.uniform(0, 2 * np.pi, n_pts)
    tube_pts = np.zeros_like(centerline_pts)

    for i in range(n_pts):
        t = tangents[i]

        # Find a vector not parallel to tangent
        if abs(t[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])

        # Gram-Schmidt orthogonalization
        perp1 = ref - np.dot(ref, t) * t
        perp1 = perp1 / np.linalg.norm(perp1)

        perp2 = np.cross(t, perp1)

        # Offset point on tube surface
        offset = tube_radius * (np.cos(angles[i]) * perp1 + np.sin(angles[i]) * perp2)
        tube_pts[i] = centerline_pts[i] + offset

    return tube_pts


def generate_microtubule_points(width, depth, numtubules, linedensity,
                              margin=0.1, spl_knots=4, spl_degree=2,
                              nudge_factor=0.1, stepsize_samples=200, return_knots=False,
                              tube_radius=None):
    from scipy.interpolate import InterpolatedUnivariateSpline
    import numpy as np

    assert spl_knots>spl_degree, "Spline degree should be smaller than number of knots"

    all_uniform_pts = []
    all_intermediate_pts = []
    all_knots = []

    while len(all_uniform_pts) < numtubules:
        # Generate endpoints within margins
        spl_ends = np.random.uniform(
            [width*margin, width*margin, 0],
            [width*(1-margin), width*(1-margin), depth],
            size=(2,3)
        )

        # Generate knot points by linear interpolation with some extra nudge (mostly straight)
        knots = np.zeros((spl_knots, 3))
        knots[0] = spl_ends[0]
        knots[-1] = spl_ends[1]

        for i in range(1, spl_knots-1):
            base_point = spl_ends[0] + (spl_ends[1]-spl_ends[0]) * i/spl_knots
            move = (np.random.rand(3)-0.5) * np.array([width,width,depth]) * nudge_factor
            knots[i] = base_point + move

        t = np.linspace(0, 1, stepsize_samples)
        pts_intermediate = np.zeros((len(t), 3))
        for i in range(3):
            spl = InterpolatedUnivariateSpline(np.arange(spl_knots), knots[:,i], k=2)
            pts_intermediate[:,i] = spl(t * (spl_knots-1))

        # Calculate step sizes and total length
        stepsizes = np.linalg.norm(np.diff(pts_intermediate, axis=0), axis=1)
        total_length = np.sum(stepsizes)
        num_points = int(total_length * linedensity)

        if num_points < 10:  # Skip if too short
            continue

        cumulative_dist = np.cumsum(stepsizes)
        desired_distances = np.linspace(0, total_length, num_points)
        t_positions = np.linspace(0, 1, len(cumulative_dist))
        dist_to_t = InterpolatedUnivariateSpline(np.append(0, cumulative_dist),
                                                np.append(0, t_positions), k=3)

        # Get uniformly spaced points
        new_t = dist_to_t(desired_distances)
        uniform_pts = np.zeros((num_points, 3))
        for dim in range(3):
            spl = InterpolatedUnivariateSpline(t, pts_intermediate[:,dim])
            uniform_pts[:,dim] = spl(new_t)

        # Project points onto tube surface if radius is specified
        if tube_radius is not None:
            uniform_pts = project_points_to_tube(uniform_pts, tube_radius)

        all_uniform_pts.append(uniform_pts)
        all_intermediate_pts.append(pts_intermediate)
        all_knots.append(knots)

    # Stack all points and knots
    final_pts = np.vstack(all_uniform_pts)
    final_intermediate = np.vstack(all_intermediate_pts)
    final_knots = np.vstack(all_knots)

    if return_knots:
        return final_pts, final_intermediate, final_knots

    return final_pts


if __name__ == '__main__':
    np.random.seed(0)
    pts, pts_1, knots = generate_microtubule_points(20, 0, 
                linedensity=40, numtubules=10, spl_knots=4, spl_degree=3, 
                tube_radius=12.5/100,
                return_knots=True)

    plt.figure()
    plt.scatter(pts[:,0],pts[:,1], s=0.4)
    #plt.scatter(pts_1[:,0],pts_1[:,1], s=4, c='r')
    plt.scatter(knots[:,0],knots[:,1], s=10, c='k')