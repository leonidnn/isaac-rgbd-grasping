// Lists Vulkan physical devices as "<uuid> <name>" without creating any logical device
// (unlike vulkaninfo, which opens a context on every GPU).
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>

int main(void) {
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "gt_vk_list";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;

    VkInstance inst;
    VkResult r = vkCreateInstance(&ci, NULL, &inst);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "vkCreateInstance failed: %d\n", r);
        return 2;
    }

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, NULL);
    VkPhysicalDevice *pd = calloc(n ? n : 1, sizeof(*pd));
    vkEnumeratePhysicalDevices(inst, &n, pd);
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceIDProperties id = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 p = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
        vkGetPhysicalDeviceProperties2(pd[i], &p);
        for (int j = 0; j < VK_UUID_SIZE; j++) printf("%02x", id.deviceUUID[j]);
        printf(" %s\n", p.properties.deviceName);
    }
    free(pd);
    vkDestroyInstance(inst, NULL);
    return 0;
}
