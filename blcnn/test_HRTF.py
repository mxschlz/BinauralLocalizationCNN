
@pytest.mark.parametrize("vertical_polar,cartesian", [
    ((0, 0, 1.4), (1.4, 0, 0)), ((90, 0, 1.4), (0, 1.4, 0)),
    ((180, 0, 1.4), (-1.4, 0, 0)), ((270, 0, 1.4), (0, -1.4, 0)),
    ((0, 90, 1.4), (0, 0, 1.4)), ((0, -90, 1.4), (0, 0, -1.4)),
    ((360, 0, 1.4), (1.4, 0, 0)), ((180, 180, 1.4), (1.4, 0, 0)),
    ((-90, 0, 1.4), (0, -1.4, 0))])
def test_eval(vertical_polar, cartesian):
    vertical_polar = np.array(vertical_polar)
    vertical_polar = vertical_polar[np.newaxis, ...]
    # print(slab.HRTF._vertical_polar_to_cartesian(vertical_polar))[0]
    np.testing.assert_array_almost_equal(slab.HRTF._vertical_polar_to_cartesian(vertical_polar)[0], cartesian)
    # assert slab.HRTF._vertical_polar_to_cartesian(vertical_polar)[0] == cartesian

def rest():
    for azim, elev in [(180, 180), (45, 0), (90, 0), (180, 0)]:
        print(f'Azim: {azim}, Elev: {elev}')
        cartesian, vertical_polar, interaural_polar = slab.HRTF._get_coordinates(np.array([azim, elev, 1.4]), 'spherical')
        print('Cartesian:', cartesian)
        print('Vertical polar:', vertical_polar)
        print('Interaural polar:', interaural_polar)
        print()

    # Plot the coordinates for fixed elevation and distance
    elev = 0
    dist = 1.4
    azims = np.arange(0, 360, 1)
    coords = np.array([[azim, elev, dist] for azim in azims])
    cartesians, vertical_polars, interaural_polars \
        = slab.HRTF._get_coordinates(coords, 'spherical')
    plt.plot(azims, cartesians[:, 0], label='x')
    plt.plot(azims, cartesians[:, 1], label='y')
    plt.plot(azims, cartesians[:, 2], label='z')
    # plt.plot(azims, vertical_polars[:, 0], label='azim')
    # plt.plot(azims, vertical_polars[:, 1], label='elev')
    # plt.plot(azims, interaural_polars[:, 0], label='int_azim')
    # plt.plot(azims, interaural_polars[:, 1], label='int_elev')
    plt.legend()
    plt.show()



# _vertical_polar_to_cartesian(vertical_polar)
# -> if we go cyclical already, then this one isn't needed
# But better do direct calculations

# _interaural_polar_to_cartesian(interaural_polar)
# _cartesian_to_vertical_polar(cartesian)
# _vertical_polar_to_interaural_polar(vertical_polar)