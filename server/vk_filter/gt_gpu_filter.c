// Vulkan layer: the application sees only the physical device whose UUID is in GT_VK_UUID
// (32 hex digits, dashes allowed). If the variable is missing or the device is not found,
// no devices are exposed at all.
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#define EXPORT __attribute__((visibility("default")))
#define MAX_OBJ 64

typedef void *key_t_;
static key_t_ key_of(const void *h) { return *(key_t_ *)h; }

typedef struct {
    key_t_ key;
    PFN_vkGetInstanceProcAddr gipa;
    PFN_vkDestroyInstance destroy;
    PFN_vkEnumeratePhysicalDevices enum_devs;
    PFN_vkEnumeratePhysicalDeviceGroups enum_groups;
    PFN_vkGetPhysicalDeviceProperties2 props2;
} inst_t;

typedef struct {
    key_t_ key;
    PFN_vkGetDeviceProcAddr gdpa;
    PFN_vkDestroyDevice destroy;
} dev_t_;

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static inst_t insts[MAX_OBJ];
static dev_t_ devs[MAX_OBJ];
static VkPhysicalDevice allowed[MAX_OBJ];
static int n_allowed;

static int want_uuid(uint8_t out[VK_UUID_SIZE]) {
    const char *s = getenv("GT_VK_UUID");
    int n = 0;
    if (!s) return 0;
    for (; *s && n < 2 * VK_UUID_SIZE; s++) {
        int v;
        if (*s == '-') continue;
        if (*s >= '0' && *s <= '9') v = *s - '0';
        else if (*s >= 'a' && *s <= 'f') v = *s - 'a' + 10;
        else if (*s >= 'A' && *s <= 'F') v = *s - 'A' + 10;
        else return 0;
        if (n % 2 == 0) out[n / 2] = (uint8_t)(v << 4);
        else out[n / 2] |= (uint8_t)v;
        n++;
    }
    return n == 2 * VK_UUID_SIZE && *s == 0;
}

static inst_t *find_inst(key_t_ k) {
    for (int i = 0; i < MAX_OBJ; i++)
        if (insts[i].key == k) return &insts[i];
    return NULL;
}

static dev_t_ *find_dev(key_t_ k) {
    for (int i = 0; i < MAX_OBJ; i++)
        if (devs[i].key == k) return &devs[i];
    return NULL;
}

static int is_ours(inst_t *in, VkPhysicalDevice pd) {
    uint8_t want[VK_UUID_SIZE];
    if (!in->props2 || !want_uuid(want)) return 0;
    VkPhysicalDeviceIDProperties id = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 p = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
    in->props2(pd, &p);
    return memcmp(id.deviceUUID, want, VK_UUID_SIZE) == 0;
}

static void remember(VkPhysicalDevice pd) {
    for (int i = 0; i < n_allowed; i++)
        if (allowed[i] == pd) return;
    if (n_allowed < MAX_OBJ) allowed[n_allowed++] = pd;
}

static int was_allowed(VkPhysicalDevice pd) {
    for (int i = 0; i < n_allowed; i++)
        if (allowed[i] == pd) return 1;
    return 0;
}

// the single matching device of this instance, or NULL
static VkPhysicalDevice our_device(inst_t *in, VkInstance instance) {
    uint32_t n = 0;
    VkPhysicalDevice found = VK_NULL_HANDLE;
    if (in->enum_devs(instance, &n, NULL) != VK_SUCCESS || n == 0) return VK_NULL_HANDLE;
    VkPhysicalDevice *all = calloc(n, sizeof(*all));
    if (!all) return VK_NULL_HANDLE;
    if (in->enum_devs(instance, &n, all) >= 0)
        for (uint32_t i = 0; i < n; i++)
            if (is_ours(in, all[i])) {
                found = all[i];
                break;
            }
    free(all);
    return found;
}

static VKAPI_ATTR VkResult VKAPI_CALL gt_CreateInstance(const VkInstanceCreateInfo *ci,
                                                        const VkAllocationCallbacks *alloc,
                                                        VkInstance *out) {
    VkLayerInstanceCreateInfo *link = (VkLayerInstanceCreateInfo *)ci->pNext;
    while (link && !(link->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
                     link->function == VK_LAYER_LINK_INFO))
        link = (VkLayerInstanceCreateInfo *)link->pNext;
    if (!link) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    PFN_vkCreateInstance create = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
    VkResult r = create(ci, alloc, out);
    if (r != VK_SUCCESS) return r;

    inst_t in = {key_of(*out), gipa};
    in.destroy = (PFN_vkDestroyInstance)gipa(*out, "vkDestroyInstance");
    in.enum_devs = (PFN_vkEnumeratePhysicalDevices)gipa(*out, "vkEnumeratePhysicalDevices");
    in.enum_groups = (PFN_vkEnumeratePhysicalDeviceGroups)gipa(*out, "vkEnumeratePhysicalDeviceGroups");
    if (!in.enum_groups)
        in.enum_groups = (PFN_vkEnumeratePhysicalDeviceGroups)gipa(*out, "vkEnumeratePhysicalDeviceGroupsKHR");
    in.props2 = (PFN_vkGetPhysicalDeviceProperties2)gipa(*out, "vkGetPhysicalDeviceProperties2");
    if (!in.props2)
        in.props2 = (PFN_vkGetPhysicalDeviceProperties2)gipa(*out, "vkGetPhysicalDeviceProperties2KHR");

    pthread_mutex_lock(&lock);
    inst_t *slot = find_inst(NULL);
    if (slot) *slot = in;
    pthread_mutex_unlock(&lock);
    if (!slot) {
        in.destroy(*out, alloc);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL gt_DestroyInstance(VkInstance instance, const VkAllocationCallbacks *alloc) {
    pthread_mutex_lock(&lock);
    inst_t *in = find_inst(key_of(instance));
    PFN_vkDestroyInstance destroy = in ? in->destroy : NULL;
    if (in) memset(in, 0, sizeof(*in));
    pthread_mutex_unlock(&lock);
    if (destroy) destroy(instance, alloc);
}

static VKAPI_ATTR VkResult VKAPI_CALL gt_EnumeratePhysicalDevices(VkInstance instance, uint32_t *count,
                                                                  VkPhysicalDevice *out) {
    pthread_mutex_lock(&lock);
    inst_t *in = find_inst(key_of(instance));
    VkPhysicalDevice pd = in ? our_device(in, instance) : VK_NULL_HANDLE;
    if (pd) remember(pd);
    pthread_mutex_unlock(&lock);

    uint32_t n = pd ? 1 : 0;
    if (!out) {
        *count = n;
        return VK_SUCCESS;
    }
    if (*count == 0 && n) return VK_INCOMPLETE;
    if (n) out[0] = pd;
    *count = n;
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL gt_EnumeratePhysicalDeviceGroups(VkInstance instance, uint32_t *count,
                                                                       VkPhysicalDeviceGroupProperties *out) {
    pthread_mutex_lock(&lock);
    inst_t *in = find_inst(key_of(instance));
    VkPhysicalDevice pd = in ? our_device(in, instance) : VK_NULL_HANDLE;
    if (pd) remember(pd);
    pthread_mutex_unlock(&lock);

    uint32_t n = pd ? 1 : 0;
    if (!out) {
        *count = n;
        return VK_SUCCESS;
    }
    if (*count == 0 && n) return VK_INCOMPLETE;
    if (n) {
        memset(out[0].physicalDevices, 0, sizeof(out[0].physicalDevices));
        out[0].physicalDeviceCount = 1;
        out[0].physicalDevices[0] = pd;
        out[0].subsetAllocation = VK_FALSE;
    }
    *count = n;
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL gt_CreateDevice(VkPhysicalDevice pd, const VkDeviceCreateInfo *ci,
                                                      const VkAllocationCallbacks *alloc, VkDevice *out) {
    pthread_mutex_lock(&lock);
    int ok = was_allowed(pd);
    pthread_mutex_unlock(&lock);
    if (!ok) return VK_ERROR_INITIALIZATION_FAILED;

    VkLayerDeviceCreateInfo *link = (VkLayerDeviceCreateInfo *)ci->pNext;
    while (link && !(link->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
                     link->function == VK_LAYER_LINK_INFO))
        link = (VkLayerDeviceCreateInfo *)link->pNext;
    if (!link) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr gdpa = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    PFN_vkCreateDevice create = (PFN_vkCreateDevice)gipa(VK_NULL_HANDLE, "vkCreateDevice");
    VkResult r = create(pd, ci, alloc, out);
    if (r != VK_SUCCESS) return r;

    dev_t_ d = {key_of(*out), gdpa, (PFN_vkDestroyDevice)gdpa(*out, "vkDestroyDevice")};
    pthread_mutex_lock(&lock);
    dev_t_ *slot = find_dev(NULL);
    if (slot) *slot = d;
    pthread_mutex_unlock(&lock);
    if (!slot) {
        d.destroy(*out, alloc);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL gt_DestroyDevice(VkDevice device, const VkAllocationCallbacks *alloc) {
    pthread_mutex_lock(&lock);
    dev_t_ *d = find_dev(key_of(device));
    PFN_vkDestroyDevice destroy = d ? d->destroy : NULL;
    if (d) memset(d, 0, sizeof(*d));
    pthread_mutex_unlock(&lock);
    if (destroy) destroy(device, alloc);
}

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL gt_GetDeviceProcAddr(VkDevice device, const char *name) {
    if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)gt_GetDeviceProcAddr;
    if (!strcmp(name, "vkDestroyDevice")) return (PFN_vkVoidFunction)gt_DestroyDevice;
    pthread_mutex_lock(&lock);
    dev_t_ *d = find_dev(key_of(device));
    PFN_vkGetDeviceProcAddr next = d ? d->gdpa : NULL;
    pthread_mutex_unlock(&lock);
    return next ? next(device, name) : NULL;
}

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL gt_GetInstanceProcAddr(VkInstance instance, const char *name) {
    static const struct {
        const char *name;
        PFN_vkVoidFunction fn;
    } ours[] = {
        {"vkGetInstanceProcAddr", (PFN_vkVoidFunction)gt_GetInstanceProcAddr},
        {"vkCreateInstance", (PFN_vkVoidFunction)gt_CreateInstance},
        {"vkDestroyInstance", (PFN_vkVoidFunction)gt_DestroyInstance},
        {"vkEnumeratePhysicalDevices", (PFN_vkVoidFunction)gt_EnumeratePhysicalDevices},
        {"vkEnumeratePhysicalDeviceGroups", (PFN_vkVoidFunction)gt_EnumeratePhysicalDeviceGroups},
        {"vkEnumeratePhysicalDeviceGroupsKHR", (PFN_vkVoidFunction)gt_EnumeratePhysicalDeviceGroups},
        {"vkCreateDevice", (PFN_vkVoidFunction)gt_CreateDevice},
        {"vkGetDeviceProcAddr", (PFN_vkVoidFunction)gt_GetDeviceProcAddr},
        {"vkDestroyDevice", (PFN_vkVoidFunction)gt_DestroyDevice},
    };
    for (size_t i = 0; i < sizeof(ours) / sizeof(ours[0]); i++)
        if (!strcmp(name, ours[i].name)) return ours[i].fn;
    if (!instance) return NULL;
    pthread_mutex_lock(&lock);
    inst_t *in = find_inst(key_of(instance));
    PFN_vkGetInstanceProcAddr next = in ? in->gipa : NULL;
    pthread_mutex_unlock(&lock);
    return next ? next(instance, name) : NULL;
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *v) {
    if (v->loaderLayerInterfaceVersion > 2) v->loaderLayerInterfaceVersion = 2;
    v->pfnGetInstanceProcAddr = gt_GetInstanceProcAddr;
    v->pfnGetDeviceProcAddr = gt_GetDeviceProcAddr;
    v->pfnGetPhysicalDeviceProcAddr = NULL;
    return VK_SUCCESS;
}
