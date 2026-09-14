#[no_mangle]
pub extern "C" fn alloc_u32(length: usize) -> *mut u32 {
    let mut values = Vec::<u32>::with_capacity(length);
    let pointer = values.as_mut_ptr();
    std::mem::forget(values);
    pointer
}

#[no_mangle]
/// # Safety
/// `pointer` must come from `alloc_u32` with the same `length` and must not have been freed.
pub unsafe extern "C" fn dealloc_u32(pointer: *mut u32, length: usize) {
    drop(Vec::from_raw_parts(pointer, 0, length));
}

#[no_mangle]
/// # Safety
/// Both pointers must reference initialized `u32` arrays of their respective lengths.
pub unsafe extern "C" fn edit_distance(
    left_pointer: *const u32,
    left_length: usize,
    right_pointer: *const u32,
    right_length: usize,
) -> usize {
    let left = std::slice::from_raw_parts(left_pointer, left_length);
    let right = std::slice::from_raw_parts(right_pointer, right_length);
    let mut previous: Vec<usize> = (0..=right_length).collect();
    let mut current = vec![0; right_length + 1];

    for (left_index, left_value) in left.iter().enumerate() {
        current[0] = left_index + 1;
        for (right_index, right_value) in right.iter().enumerate() {
            let substitution = previous[right_index] + usize::from(left_value != right_value);
            let insertion = current[right_index] + 1;
            let deletion = previous[right_index + 1] + 1;
            current[right_index + 1] = substitution.min(insertion).min(deletion);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    previous[right_length]
}
